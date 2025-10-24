import gradio as gr
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import os
import shutil
import re
import json
import random
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import io


# #############################################################################
# CORE LOGIC FROM UTILITY SCRIPTS
# #############################################################################

# --- From organise_dataset.py ---
def util_get_class_name_from_filename(filename):
    match = re.match(r'(.+?)_\d{3,}', filename)
    return match.group(1) if match else None

def util_create_class_mapping(train_txt_path, log_capture):
    class_mapping = {}
    try:
        with open(train_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, class_id_str = parts
                    class_id = int(class_id_str)
                    if class_id not in class_mapping:
                        class_name = util_get_class_name_from_filename(filename)
                        if class_name:
                            class_mapping[class_id] = class_name
    except FileNotFoundError:
        print(f"Error: {train_txt_path} not found.", file=log_capture)
        return None
    return class_mapping

def util_process_dataset(base_dir, annotations_file, source_subdir, dest_dir, class_mapping, log_capture):
    annotations_path = os.path.join(base_dir, annotations_file)
    source_dir = os.path.join(base_dir, source_subdir)
    if not os.path.exists(annotations_path):
        print(f"Warning: Annotations file not found: {annotations_path}", file=log_capture)
        return
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory not found: {source_dir}", file=log_capture)
        return
    with open(annotations_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, class_id_str = parts
                class_id = int(class_id_str)
                if class_id in class_mapping:
                    class_name = class_mapping[class_id]
                    class_dir = os.path.join(dest_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    source_path = os.path.join(source_dir, filename)
                    dest_path = os.path.join(class_dir, filename)
                    if os.path.exists(source_path):
                        print(f"Moving {source_path} to {dest_path}", file=log_capture)
                        shutil.move(source_path, dest_path)
                    else:
                        print(f"Warning: Source file not found: {source_path}", file=log_capture)
                else:
                    print(f"Warning: Class ID {class_id} for file {filename} not in mapping.", file=log_capture)

# --- From normalise_class_names.py ---
def util_normalise_class_names(target_dir, log_capture):
    print(f"\nStarting normalisation in '{target_dir}'...", file=log_capture)
    try:
        subdirectories = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
        if not subdirectories:
            print("No subdirectories found to normalise.", file=log_capture)
            return
        for old_name in subdirectories:
            new_name = old_name.lower()
            if old_name == new_name:
                print(f"Skipping '{old_name}' as it is already normalised.", file=log_capture)
                continue
            old_path = os.path.join(target_dir, old_name)
            new_path = os.path.join(target_dir, new_name)
            if os.path.exists(new_path) and not os.path.samefile(old_path, new_path):
                print(f"Warning: Cannot rename '{old_name}' to '{new_name}' because a different directory with that name already exists. Skipping.", file=log_capture)
                continue
            try:
                temp_name = old_name + "_temp_rename"
                temp_path = os.path.join(target_dir, temp_name)
                if os.path.exists(temp_path):
                    print(f"Warning: Temporary path '{temp_path}' already exists. Skipping rename for '{old_name}'.", file=log_capture)
                    continue
                os.rename(old_path, temp_path)
                os.rename(temp_path, new_path)
                print(f"Renamed '{old_name}' to '{new_name}'.", file=log_capture)
            except OSError as e:
                print(f"Error renaming '{old_name}': {e}", file=log_capture)
        print("\nNormalisation complete.", file=log_capture)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=log_capture)

# --- From normalise_image_names.py ---
def util_lowercase_filenames(target_dir, log_capture):
    file_count, renamed_count = 0, 0
    for dirpath, _, filenames in os.walk(target_dir):
        for old_name in filenames:
            file_count += 1
            new_name = old_name.lower()
            if old_name == new_name: continue
            old_path, new_path = os.path.join(dirpath, old_name), os.path.join(dirpath, new_name)
            if os.path.exists(new_path) and not os.path.samefile(old_path, new_path):
                print(f"Warning: Cannot rename '{old_path}' to '{new_path}'. Skipping.", file=log_capture)
                continue
            try:
                temp_name = old_name + "_temp_rename"
                temp_path = os.path.join(dirpath, temp_name)
                if os.path.exists(temp_path):
                    print(f"Warning: Temp path '{temp_path}' exists. Skipping '{old_name}'.", file=log_capture)
                    continue
                os.rename(old_path, temp_path)
                os.rename(temp_path, new_path)
                print(f"Renamed '{os.path.basename(old_path)}' to '{os.path.basename(new_path)}'", file=log_capture)
                renamed_count += 1
            except OSError as e:
                print(f"Error renaming '{old_path}': {e}", file=log_capture)
    print(f"\nProcessed {file_count} files, renamed {renamed_count}.", file=log_capture)

def util_standardise_filenames(target_dir, to_lowercase, log_capture):
    total_renamed = 0
    for dirpath, _, filenames in os.walk(target_dir):
        if not filenames: continue
        class_name = os.path.basename(dirpath).lower() if to_lowercase else os.path.basename(dirpath)
        print(f"\nProcessing directory: {class_name}", file=log_capture)
        rename_map = []
        for i, old_name in enumerate(sorted(filenames)):
            _, extension = os.path.splitext(old_name)
            if to_lowercase: extension = extension.lower()
            new_name = f"{class_name}_{i+1:04d}{extension}"
            rename_map.append({'old_path': os.path.join(dirpath, old_name), 'new_path': os.path.join(dirpath, new_name)})
        for item in rename_map:
            item['temp_path'] = item['old_path'] + '.tmp'
            os.rename(item['old_path'], item['temp_path'])
        for item in rename_map:
            os.rename(item['temp_path'], item['new_path'])
            print(f"Renamed '{os.path.basename(item['old_path'])}' to '{os.path.basename(item['new_path'])}'", file=log_capture)
            total_renamed += 1
    print(f"\nStandardised {total_renamed} files.", file=log_capture)

# --- From autotrain_dataset_splitter.py ---
def util_split_image_dataset(source_dir, output_dir, min_images_per_split, log_capture):
    train_dir, validation_dir = f"{output_dir}_train", f"{output_dir}_validation"
    if os.path.exists(train_dir): shutil.rmtree(train_dir)
    if os.path.exists(validation_dir): shutil.rmtree(validation_dir)
    class_dirs = [r for r, d, f in os.walk(source_dir) if not d and f]
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    required_total = min_images_per_split * 2
    class_image_data = {cd: [f for f in os.listdir(cd) if os.path.splitext(f)[1].lower() in image_extensions] for cd in class_dirs}
    valid_class_count = sum(1 for images in class_image_data.values() if len(images) >= required_total)
    if valid_class_count < 2:
        print(f"Error: Dataset splitting requires at least 2 classes with >= {required_total} images each. Found {valid_class_count} valid classes.", file=log_capture)
        return
    manifest = {"included_classes": {}, "skipped_classes": {}}
    processed_class_names = set()
    for class_dir, images in class_image_data.items():
        base_class_name = os.path.basename(class_dir)
        class_name, counter = base_class_name, 1
        while class_name in processed_class_names:
            class_name = f"{base_class_name}_{counter}"; counter += 1
        processed_class_names.add(class_name)
        if len(images) < required_total:
            manifest["skipped_classes"][class_name] = {"count": len(images), "reason": f"Not enough images ({len(images)}), required {required_total}."}
            continue
        random.shuffle(images)
        num_val_ratio = round(len(images) * 0.2)
        num_train_ratio = len(images) - num_val_ratio
        num_validation = num_val_ratio if num_val_ratio >= min_images_per_split and num_train_ratio >= min_images_per_split else min_images_per_split
        validation_images, train_images = images[:num_validation], images[num_validation:]
        manifest["included_classes"][class_name] = {"train": len(train_images), "validation": len(validation_images)}
        for split_dir, split_images in [(train_dir, train_images), (validation_dir, validation_images)]:
            split_class_dir = os.path.join(split_dir, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for image in split_images:
                shutil.copy(os.path.join(class_dir, image), os.path.join(split_class_dir, image))
    manifest_path = f"{output_dir}_manifest.json"
    with open(manifest_path, "w") as f: json.dump(manifest, f, indent=4)
    print(f"Dataset split complete. Manifest: {manifest_path}", file=log_capture)

# --- From directory_manifest.py ---
def util_generate_manifest(directory, manifest_path, log_capture):
    ignored_dirs = {'.git', '__pycache__', '.vscode', '.idea', 'node_modules', 'venv', '.venv'}
    ignored_files = {os.path.basename(manifest_path)}
    ignored_extensions = {'.pyc', '.zip', '.log', '.tmp', '.bak', '.swp'}
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            for root, dirs, files in os.walk(directory, topdown=True):
                dirs[:] = [d for d in dirs if d not in ignored_dirs]
                for filename in files:
                    if filename in ignored_files or os.path.splitext(filename)[1].lower() in ignored_extensions:
                        continue
                    relative_path = os.path.relpath(os.path.join(root, filename), directory).replace(os.sep, '/')
                    f.write(f"- {relative_path}\n")
        print(f"Manifest file created at: {manifest_path}", file=log_capture)
    except Exception as e:
        print(f"An error occurred: {e}", file=log_capture)

# --- From check_balance.py ---
def util_get_class_from_line(line: str):
    return line.strip().lstrip('- ').split('/')[0] if '/' in line else None

def util_analyse_balance(manifest_path, log_capture):
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at '{manifest_path}'", file=log_capture)
        return None, None
    with open(manifest_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    class_counts = Counter(c for line in lines if (c := util_get_class_from_line(line)))
    if not class_counts:
        print("No classes found in the manifest file.", file=log_capture)
        return "No classes found.", None
    counts = list(class_counts.values())
    imbalance_ratio = max(counts) / min(counts)
    summary = (
        f"Dataset Balance Analysis\n"
        f"=========================\n"
        f"Total classes: {len(class_counts)}\n"
        f"Total images: {sum(counts)}\n"
        f"Images per class:\n"
        f"  - Minimum: {min(counts)}\n"
        f"  - Maximum: {max(counts)}\n"
        f"  - Average: {np.mean(counts):.2f}\n"
        f"  - Std Dev: {np.std(counts):.2f}\n"
        f"Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}:1"
    )
    print(summary, file=log_capture)
    sorted_classes = sorted(class_counts.keys())
    sorted_counts = [class_counts[c] for c in sorted_classes]
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(sorted_classes, sorted_counts)
    ax.set_xlabel('Class'); ax.set_ylabel('Number of Images'); ax.set_title('Image Distribution Across Classes')
    plt.xticks(rotation=90, fontsize='small'); plt.tight_layout()
    return fig

# --- From count_classes.py ---
def util_count_classes(target_dir, save_to_manifest, manifest_path, log_capture):
    if not os.path.isdir(target_dir):
        print(f"Error: Directory not found at '{target_dir}'", file=log_capture)
        return
    try:
        class_dirs = [e for e in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, e))]
        if not class_dirs:
            print(f"No class subdirectories found in '{target_dir}'.", file=log_capture)
            return
        class_counts = Counter({name: len([f for f in os.listdir(os.path.join(target_dir, name)) if os.path.isfile(os.path.join(target_dir, name, f))]) for name in class_dirs})
        sorted_counts = sorted(class_counts.items())
        print(f"Found {len(class_dirs)} classes.", file=log_capture)
        print("-" * 20, file=log_capture)
        for class_name, count in sorted_counts: print(f"{class_name}: {count} items", file=log_capture)
        print("-" * 20, file=log_capture)
        if save_to_manifest:
            with open(manifest_path, 'w') as f:
                f.write(f"# Class Count Manifest\n\n**Total classes:** {len(class_dirs)}\n\n| Class Name | Item Count |\n|---|---|\n")
                for class_name, count in sorted_counts: f.write(f"| {class_name} | {count} |\n")
            print(f"Manifest saved to {manifest_path}", file=log_capture)
    except OSError as e:
        print(f"Error accessing directory '{target_dir}': {e}", file=log_capture)

# --- From plot_metrics.py ---
def util_plot_training_metrics(json_path):
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    df = pd.DataFrame(data.get('log_history', []))
    if df.empty: raise ValueError("No 'log_history' found.")
    train_df = df[df['loss'].notna()].copy()
    eval_df = df[df['eval_loss'].notna()].copy()
    figures = {}
    # Plot Loss
    fig_loss, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Training vs. Evaluation Loss')
    if 'loss' in train_df: ax.plot(train_df['step'], train_df['loss'], label='Training Loss', marker='o')
    if 'eval_loss' in eval_df: ax.plot(eval_df['step'], eval_df['eval_loss'], label='Evaluation Loss', marker='x')
    ax.legend(); ax.grid(True); figures['Loss'] = fig_loss
    # Plot Accuracy
    fig_acc, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Accuracy')
    if 'eval_accuracy' in eval_df: ax.plot(eval_df['step'], eval_df['eval_accuracy'], label='Evaluation Accuracy', marker='o', color='g')
    ax.legend(); ax.grid(True); figures['Accuracy'] = fig_acc
    # Plot Learning Rate
    fig_lr, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Learning Rate Schedule')
    if 'learning_rate' in train_df: ax.plot(train_df['step'], train_df['learning_rate'], label='Learning Rate', marker='o', color='r')
    ax.legend(); ax.grid(True); figures['Learning Rate'] = fig_lr
    # Plot Grad Norm
    fig_gn, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Gradient Norm')
    if 'grad_norm' in train_df: ax.plot(train_df['step'], train_df['grad_norm'], label='Grad Norm', marker='o', color='purple')
    ax.legend(); ax.grid(True); figures['Gradient Norm'] = fig_gn
    # Plot F1
    fig_f1, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation F1 Scores')
    if 'eval_f1_macro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_f1_macro'], label='F1 Macro', marker='o')
    if 'eval_f1_micro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_f1_micro'], label='F1 Micro', marker='x')
    if 'eval_f1_weighted' in eval_df: ax.plot(eval_df['step'], eval_df['eval_f1_weighted'], label='F1 Weighted', marker='s')
    ax.legend(); ax.grid(True); figures['F1 Scores'] = fig_f1
    # Plot Precision
    fig_prec, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Precision Scores')
    if 'eval_precision_macro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_precision_macro'], label='Precision Macro', marker='o')
    if 'eval_precision_micro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_precision_micro'], label='Precision Micro', marker='x')
    if 'eval_precision_weighted' in eval_df: ax.plot(eval_df['step'], eval_df['eval_precision_weighted'], label='Precision Weighted', marker='s')
    ax.legend(); ax.grid(True); figures['Precision'] = fig_prec
    # Plot Recall
    fig_recall, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Recall Scores')
    if 'eval_recall_macro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_recall_macro'], label='Recall Macro', marker='o')
    if 'eval_recall_micro' in eval_df: ax.plot(eval_df['step'], eval_df['eval_recall_micro'], label='Recall Micro', marker='x')
    if 'eval_recall_weighted' in eval_df: ax.plot(eval_df['step'], eval_df['eval_recall_weighted'], label='Recall Weighted', marker='s')
    ax.legend(); ax.grid(True); figures['Recall'] = fig_recall
    # Plot Epoch
    fig_epoch, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Epoch Progression')
    if 'epoch' in df:
        epoch_df = df[['step', 'epoch']].dropna().drop_duplicates('step').sort_values('step')
        ax.plot(epoch_df['step'], epoch_df['epoch'], label='Epoch', marker='.')
    ax.legend(); ax.grid(True); figures['Epoch'] = fig_epoch
    # Plot Eval Runtime
    fig_runtime, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Runtime')
    if 'eval_runtime' in eval_df: ax.plot(eval_df['step'], eval_df['eval_runtime'], label='Eval Runtime', marker='o')
    ax.legend(); ax.grid(True); figures['Eval Runtime'] = fig_runtime
    # Plot Eval Samples Per Second
    fig_sps, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Samples Per Second')
    if 'eval_samples_per_second' in eval_df: ax.plot(eval_df['step'], eval_df['eval_samples_per_second'], label='Eval Samples/sec', marker='o')
    ax.legend(); ax.grid(True); figures['Eval Samples/sec'] = fig_sps
    # Plot Eval Steps Per Second
    fig_steps_ps, ax = plt.subplots(figsize=(10, 6)); ax.set_title('Evaluation Steps Per Second')
    if 'eval_steps_per_second' in eval_df: ax.plot(eval_df['step'], eval_df['eval_steps_per_second'], label='Eval Steps/sec', marker='o')
    ax.legend(); ax.grid(True); figures['Eval Steps/sec'] = fig_steps_ps
    return figures

# #############################################################################
# GRADIO WRAPPER FUNCTIONS
# #############################################################################

def classify_bird(model_path: str, input_image: Image.Image) -> dict:
    if not model_path:
        raise gr.Error("Please select a model directory.")

    model_dir = model_path
    if os.path.isfile(model_path):
        model_dir = os.path.dirname(model_path)

    try:
        image_processor = AutoImageProcessor.from_pretrained(model_dir)
        model = AutoModelForImageClassification.from_pretrained(model_dir)
    except Exception as e:
        raise gr.Error(f"Error loading model from {model_dir}. Check path and files. Original error: {e}")
    inputs = image_processor(images=input_image, return_tensors="pt")
    with torch.no_grad(): outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    return {model.config.id2label[i.item()]: p.item() for i, p in zip(top5_indices, top5_prob)}

def run_with_log_capture(func, *args):
    log_capture = io.StringIO()
    with redirect_stdout(log_capture):
        func(*args, log_capture=log_capture)
    return log_capture.getvalue()

def run_organise_dataset(base_dir):
    log_capture = io.StringIO()
    with redirect_stdout(log_capture):
        processed_dir = os.path.join(base_dir, 'Processed_Dataset')
        os.makedirs(processed_dir, exist_ok=True)
        train_txt_path = os.path.join(base_dir, 'train.txt')
        class_mapping = util_create_class_mapping(train_txt_path, log_capture)
        if not class_mapping:
            print("Error: Could not create class mapping. Aborting.", file=log_capture)
            return log_capture.getvalue()
        print("Processing training set...", file=log_capture)
        util_process_dataset(base_dir, 'train.txt', 'Train', processed_dir, class_mapping, log_capture)
        print("\nProcessing test set...", file=log_capture)
        util_process_dataset(base_dir, 'test.txt', 'Test', processed_dir, class_mapping, log_capture)
        print("\nDataset organisation complete.", file=log_capture)
    return log_capture.getvalue()

def run_normalise_class_names(target_dir):
    return run_with_log_capture(util_normalise_class_names, target_dir)

def run_normalise_image_names(target_dir, to_lowercase, to_standardise):
    log_capture = io.StringIO()
    with redirect_stdout(log_capture):
        if to_standardise:
            print("Standardising filenames...", file=log_capture)
            util_standardise_filenames(target_dir, to_lowercase, log_capture)
        elif to_lowercase:
            print("Converting filenames to lowercase...", file=log_capture)
            util_lowercase_filenames(target_dir, log_capture)
    return log_capture.getvalue()

def run_split_dataset(source_dir, output_dir, min_images):
    return run_with_log_capture(util_split_image_dataset, source_dir, output_dir, min_images)

def run_generate_manifest(directory, manifest_path):
    return run_with_log_capture(util_generate_manifest, directory, manifest_path)

def run_check_balance(manifest_path):
    log_capture = io.StringIO()
    with redirect_stdout(log_capture):
        fig = util_analyse_balance(manifest_path, log_capture)
    return log_capture.getvalue(), fig

def run_count_classes(target_dir, save_to_manifest, manifest_path):
    return run_with_log_capture(util_count_classes, target_dir, save_to_manifest, manifest_path)

def show_model_charts(model_dir):
    """Finds trainer_state.json, returns metric plots, and the model_dir for sync."""
    if not model_dir:
        return (None,) * 11 + (gr.update(visible=False), None)

    json_path = None
    for root, _, files in os.walk(model_dir):
        if 'trainer_state.json' in files:
            json_path = os.path.join(root, 'trainer_state.json')
            break

    if not json_path:
        print(f"trainer_state.json not found in {model_dir}")
        return (None,) * 11 + (gr.update(visible=False), model_dir)

    try:
        figures = util_plot_training_metrics(json_path)
        return (
            figures.get('Loss'), figures.get('Accuracy'), figures.get('Learning Rate'),
            figures.get('Gradient Norm'), figures.get('F1 Scores'), figures.get('Precision'),
            figures.get('Recall'), figures.get('Epoch'), figures.get('Eval Runtime'),
            figures.get('Eval Samples/sec'), figures.get('Eval Steps/sec'),
            gr.update(visible=True),
            model_dir
        )
    except Exception as e:
        print(f"Error generating plots for {json_path}: {e}")
        return (None,) * 11 + (gr.update(visible=False), model_dir)


def run_plot_metrics(json_path):
    try:
        figures = util_plot_training_metrics(json_path)
        return (
            figures.get('Loss'), figures.get('Accuracy'), figures.get('Learning Rate'),
            figures.get('Gradient Norm'), figures.get('F1 Scores'), figures.get('Precision'),
            figures.get('Recall'), figures.get('Epoch'), figures.get('Eval Runtime'),
            figures.get('Eval Samples/sec'), figures.get('Eval Steps/sec')
        )
    except Exception as e:
        raise gr.Error(str(e))

def get_model_choices():
    """Returns a list of directories in the current directory that start with 'Model-'."""
    try:
        return [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('Model-')]
    except FileNotFoundError:
        print("Warning: Could not find the current directory to scan for models.")
        return []

def update_model_choices():
    """Refreshes the list of available models in the dropdowns."""
    choices = get_model_choices()
    return gr.update(choices=choices), gr.update(choices=choices)

# #############################################################################
# GRADIO UI
# #############################################################################

with gr.Blocks(theme=gr.themes.Monochrome(), title="Multi-Class Classification (Bird Species)") as demo:
    gr.Markdown("# Multi-Class Classification (Bird Species)")

    gr.HTML('<script>setInterval(() => { const btn = document.getElementById("model_refresh_button"); if (btn) { btn.click(); } }, 5000)</script>', visible=False)
    refresh_button = gr.Button(elem_id="model_refresh_button", visible=False)

    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column(scale=1):
                inf_model_path = gr.Dropdown(label="Select Model", choices=get_model_choices(), value=None)
                inf_input_image = gr.Image(type="pil", label="Upload a bird image")
            with gr.Column(scale=1):
                inf_output_label = gr.Label(num_top_classes=5, label="Predictions")
                inf_button = gr.Button("Classify", variant="primary")
        inf_button.click(classify_bird, inputs=[inf_model_path, inf_input_image], outputs=inf_output_label)

    with gr.Tab("Training Metrics"):
        metrics_model_path = gr.Dropdown(label="Select Model", choices=get_model_choices(), value=None)
        with gr.Column(visible=False) as inf_plots_container:
            with gr.Row():
                inf_plot_loss = gr.Plot(label="Loss")
                inf_plot_acc = gr.Plot(label="Accuracy")
            with gr.Row():
                inf_plot_lr = gr.Plot(label="Learning Rate")
                inf_plot_grad = gr.Plot(label="Gradient Norm")
            with gr.Row():
                inf_plot_f1 = gr.Plot(label="F1 Scores")
                inf_plot_prec = gr.Plot(label="Precision")
            with gr.Row():
                inf_plot_recall = gr.Plot(label="Recall")
                inf_plot_epoch = gr.Plot(label="Epoch")
            with gr.Row():
                inf_plot_runtime = gr.Plot(label="Eval Runtime")
                inf_plot_sps = gr.Plot(label="Eval Samples/sec")
            with gr.Row():
                inf_plot_steps_ps = gr.Plot(label="Eval Steps/sec")

        inf_plots = [
            inf_plot_loss, inf_plot_acc, inf_plot_lr, inf_plot_grad, inf_plot_f1,
            inf_plot_prec, inf_plot_recall, inf_plot_epoch, inf_plot_runtime,
            inf_plot_sps, inf_plot_steps_ps
        ]
        inf_model_path.change(
            fn=show_model_charts,
            inputs=[inf_model_path],
            outputs=inf_plots + [inf_plots_container, metrics_model_path]
        )
        metrics_model_path.change(
            fn=show_model_charts,
            inputs=[metrics_model_path],
            outputs=inf_plots + [inf_plots_container, inf_model_path]
        )
    with gr.Tab("Data Preparation"):
        gr.Markdown("## Tools for Preparing Your Dataset")
        with gr.Accordion("1. Organise Raw Dataset", open=False):
            gr.Markdown("Organises a raw dataset (like CUB_200_2011) into a structured format. It reads `train.txt` and `test.txt` to move images from `Train/` and `Test/` subdirectories into class-specific folders inside a `Processed_Dataset` directory.")
            prep_org_basedir = gr.Textbox(label="Base Dataset Directory", placeholder="e.g., 'C:/Users/Me/Downloads/CUB_200_2011'")
            prep_org_button = gr.Button("Organise Dataset")
            prep_org_log = gr.Textbox(label="Log", interactive=False, lines=10)
            prep_org_button.click(run_organise_dataset, inputs=[prep_org_basedir], outputs=prep_org_log)
        with gr.Accordion("2. Normalise Class Directory Names", open=False):
            gr.Markdown("Renames all class subdirectories within a target directory to be lowercase. This helps ensure consistency, which is important for many training frameworks.")
            prep_norm_class_dir = gr.Textbox(label="Target Directory", placeholder="e.g., 'C:/.../Processed_Dataset'")
            prep_norm_class_button = gr.Button("Normalise Class Names")
            prep_norm_class_log = gr.Textbox(label="Log", interactive=False, lines=10)
            prep_norm_class_button.click(run_normalise_class_names, inputs=[prep_norm_class_dir], outputs=prep_norm_class_log)
        with gr.Accordion("3. Normalise Image Filenames", open=False):
            gr.Markdown("Processes image filenames within a directory. It can convert all filenames to lowercase and/or standardise them into a `class_name_xxxx.ext` format. This is useful for cleaning up dataset naming conventions.")
            prep_norm_img_dir = gr.Textbox(label="Target Directory", placeholder="e.g., 'C:/.../Processed_Dataset'")
            prep_norm_img_lower = gr.Checkbox(label="Convert filenames to lowercase", value=True)
            prep_norm_img_std = gr.Checkbox(label="Standardise filenames (e.g., class_0001.jpg)")
            prep_norm_img_button = gr.Button("Process Image Names")
            prep_norm_img_log = gr.Textbox(label="Log", interactive=False, lines=10)
            prep_norm_img_button.click(run_normalise_image_names, inputs=[prep_norm_img_dir, prep_norm_img_lower, prep_norm_img_std], outputs=prep_norm_img_log)
        with gr.Accordion("4. Split Dataset for AutoTrain", open=False):
            gr.Markdown("Splits a structured dataset into `training` and `validation` sets, suitable for use with tools like AutoTrain. It ensures that each class has a minimum number of images in both splits and creates a manifest file detailing the results.")
            prep_split_source = gr.Textbox(label="Source Directory", placeholder="e.g., 'C:/.../Processed_Dataset'")
            prep_split_output = gr.Textbox(label="Output Directory Name", placeholder="e.g., 'autotrain_dataset'")
            prep_split_min = gr.Number(label="Min Images Per Split", value=5)
            prep_split_button = gr.Button("Split Dataset")
            prep_split_log = gr.Textbox(label="Log", interactive=False, lines=10)
            prep_split_button.click(run_split_dataset, inputs=[prep_split_source, prep_split_output, prep_split_min], outputs=prep_split_log)

    with gr.Tab("Analysis & Utilities"):
        gr.Markdown("## Tools for Analysis and File Management")
        with gr.Accordion("Check Dataset Balance", open=False):
            gr.Markdown("Analyses a dataset manifest file to check for class imbalance. It provides a summary of image counts per class and generates a bar chart to visualise the distribution.")
            analysis_balance_path = gr.Textbox(label="Path to Manifest File")
            analysis_balance_button = gr.Button("Analyse Balance")
            analysis_balance_log = gr.Textbox(label="Summary", interactive=False, lines=10)
            analysis_balance_plot = gr.Plot(label="Class Distribution")
            analysis_balance_button.click(run_check_balance, inputs=[analysis_balance_path], outputs=[analysis_balance_log, analysis_balance_plot])
        with gr.Accordion("Count Classes in Directory", open=False):
            gr.Markdown("Counts the number of subdirectories (classes) and the number of files (items) within each class in a given directory. It can optionally save this information to a markdown manifest file.")
            util_count_dir = gr.Textbox(label="Dataset Directory")
            util_count_save = gr.Checkbox(label="Save to manifest file")
            util_count_path = gr.Textbox(label="Manifest File Path", value="class_counts.md")
            util_count_button = gr.Button("Count Classes")
            util_count_log = gr.Textbox(label="Log", interactive=False, lines=10)
            util_count_button.click(run_count_classes, inputs=[util_count_dir, util_count_save, util_count_path], outputs=util_count_log)
        with gr.Accordion("Generate Directory Manifest", open=False):
            gr.Markdown("Creates a manifest file listing all files within a specified directory and its subdirectories. It's useful for getting an overview of a project's structure or for creating file lists for other processes.")
            util_manifest_dir = gr.Textbox(label="Target Directory")
            util_manifest_path = gr.Textbox(label="Save Manifest As", value="manifest.md")
            util_manifest_button = gr.Button("Generate Manifest")
            util_manifest_log = gr.Textbox(label="Log", interactive=False, lines=5)
            util_manifest_button.click(run_generate_manifest, inputs=[util_manifest_dir, util_manifest_path], outputs=util_manifest_log)

    refresh_button.click(
        fn=update_model_choices,
        inputs=[],
        outputs=[inf_model_path, metrics_model_path]
    )
    demo.load(
        fn=update_model_choices,
        inputs=[],
        outputs=[inf_model_path, metrics_model_path]
    )

if __name__ == "__main__":
    demo.launch()
