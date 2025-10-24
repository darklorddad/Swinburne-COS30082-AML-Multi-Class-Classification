import gradio as gr
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import os
import shutil
from contextlib import redirect_stdout
import io
import zipfile
import tempfile
import matplotlib.pyplot as plt
import subprocess
import sys
import webbrowser
import time
import requests

from utils import (
    util_create_class_mapping, util_process_dataset, util_normalise_class_names,
    util_standardise_filenames, util_lowercase_filenames, util_split_image_dataset,
    util_generate_manifest, util_analyse_balance, util_count_classes,
    util_plot_training_metrics
)

AUTOTRAIN_PROCESS = None


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

def run_organise_dataset(train_zip_file, test_zip_file, train_txt_file, test_txt_file, output_dir):
    log_capture = io.StringIO()
    with redirect_stdout(log_capture):
        if not all([train_zip_file, test_zip_file, train_txt_file, test_txt_file, output_dir]):
            print("Error: Please provide all required files and the output directory.", file=log_capture)
            return log_capture.getvalue()

        train_zip_path = train_zip_file.name
        test_zip_path = test_zip_file.name
        train_txt_path = train_txt_file.name
        test_txt_path = test_txt_file.name
        output_dir_path = output_dir

        def extract_zip_and_get_basedir(zip_path, prefix, log_stream):
            if not os.path.isfile(zip_path) or not zipfile.is_zipfile(zip_path):
                print(f"Error: {zip_path} is not a valid zip file.", file=log_stream)
                return None, None
            
            temp_dir = tempfile.mkdtemp(prefix=prefix)
            print(f"Extracting {os.path.basename(zip_path)} to {temp_dir}", file=log_stream)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            extracted_items = os.listdir(temp_dir)
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_dir, extracted_items[0])):
                base_dir = os.path.join(temp_dir, extracted_items[0])
                print(f"Using extracted sub-directory as base: {base_dir}", file=log_stream)
            else:
                base_dir = temp_dir
            return base_dir, temp_dir

        train_temp_base_dir, train_temp_root = None, None
        test_temp_base_dir, test_temp_root = None, None

        try:
            train_temp_base_dir, train_temp_root = extract_zip_and_get_basedir(train_zip_path, "autotrain_train_", log_capture)
            if not train_temp_base_dir:
                return log_capture.getvalue()

            test_temp_base_dir, test_temp_root = extract_zip_and_get_basedir(test_zip_path, "autotrain_test_", log_capture)
            if not test_temp_base_dir:
                return log_capture.getvalue()

            os.makedirs(output_dir_path, exist_ok=True)
            print(f"Processed dataset will be saved to: {output_dir_path}", file=log_capture)

            class_mapping = util_create_class_mapping(train_txt_path, log_capture)
            if not class_mapping:
                print("Error: Could not create class mapping from train.txt. Aborting.", file=log_capture)
                return log_capture.getvalue()

            print("\nProcessing training set...", file=log_capture)
            util_process_dataset(train_txt_path, train_temp_base_dir, output_dir_path, class_mapping, log_capture)

            print("\nProcessing test set...", file=log_capture)
            util_process_dataset(test_txt_path, test_temp_base_dir, output_dir_path, class_mapping, log_capture)

            print("\nDataset organisation complete.", file=log_capture)
        finally:
            if train_temp_root:
                print(f"Cleaning up temporary directory: {train_temp_root}", file=log_capture)
                shutil.rmtree(train_temp_root)
            if test_temp_root:
                print(f"Cleaning up temporary directory: {test_temp_root}", file=log_capture)
                shutil.rmtree(test_temp_root)
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

def run_generate_manifest(directory, save_manifest, manifest_path):
    log_capture = io.StringIO()
    with redirect_stdout(log_capture):
        try:
            directory_path = directory.name if hasattr(directory, 'name') else directory
            util_generate_manifest(directory_path, save_manifest, manifest_path, log_capture)
        except Exception as e:
            print(f"An error occurred: {e}", file=log_capture)
    return log_capture.getvalue()

def run_check_balance(manifest_path):
    try:
        summary, fig = util_analyse_balance(manifest_path)
        if fig is None:
            return summary, None, None, None, gr.update(visible=False)
        return summary, fig, summary, fig, gr.update(visible=True)
    except Exception as e:
        return str(e), None, None, None, gr.update(visible=False)

def save_balance_analysis(summary, fig, output_basename):
    if not summary or fig is None or not output_basename:
        return "Nothing to save or invalid basename."
    
    log_capture = io.StringIO()
    with redirect_stdout(log_capture):
        try:
            summary_path = f"{output_basename}_summary.txt"
            plot_path = f"{output_basename}_plot.png"
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Summary saved to {summary_path}", file=log_capture)
            
            fig.savefig(plot_path)
            plt.close(fig) # Close the figure to free up memory
            print(f"Plot saved to {plot_path}", file=log_capture)
            
        except Exception as e:
            print(f"Error saving analysis: {e}", file=log_capture)
            
    return log_capture.getvalue()

def run_count_classes(target_dir, save_to_manifest, manifest_path):
    return run_with_log_capture(util_count_classes, target_dir, save_to_manifest, manifest_path)

def launch_autotrain_ui():
    """Launches the AutoTrain Gradio UI and opens it in a new browser tab."""
    global AUTOTRAIN_PROCESS
    command = [sys.executable, "launch_autotrain.py"]
    autotrain_url = "http://localhost:7861"
    try:
        # Redirect stdout/stderr to prevent blocking and hide console window on Windows
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        AUTOTRAIN_PROCESS = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo
        )
        
        # Poll for the server to be ready
        start_time = time.time()
        timeout = 30  # seconds
        server_ready = False
        
        print("Waiting for AutoTrain UI to start...")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(autotrain_url, timeout=1)
                if response.status_code == 200:
                    print("AutoTrain UI is ready.")
                    server_ready = True
                    break
            except requests.ConnectionError:
                time.sleep(1)
            except requests.Timeout:
                pass # Ignore timeouts and continue polling
        
        if server_ready:
            webbrowser.open(autotrain_url)
            message = f"Successfully launched AutoTrain UI. It should now be open in your web browser at {autotrain_url}."
            print(message)
            return message, gr.update(visible=False), gr.update(visible=True)
        else:
            # Server failed to start within timeout, so we should stop the zombie process.
            stop_autotrain_ui()
            message = f"AutoTrain UI failed to start within {timeout} seconds. The process has been stopped."
            print(message)
            return message, gr.update(visible=True), gr.update(visible=False)

    except Exception as e:
        message = f"Failed to launch AutoTrain UI: {e}"
        print(message)
        return message, gr.update(visible=True), gr.update(visible=False)

def stop_autotrain_ui():
    """Stops the AutoTrain UI process."""
    global AUTOTRAIN_PROCESS
    process = AUTOTRAIN_PROCESS
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
            message = "AutoTrain UI process has been stopped."
        except subprocess.TimeoutExpired:
            process.kill()
            message = "AutoTrain UI process did not stop gracefully and was killed."
        except Exception as e:
            message = f"Error stopping AutoTrain UI: {e}"
            print(message)
            return message, gr.update(visible=False), gr.update(visible=True)
        
        print(message)
        AUTOTRAIN_PROCESS = None
        return message, gr.update(visible=True), gr.update(visible=False)
    else:
        message = "AutoTrain UI process is not running or was already stopped."
        print(message)
        AUTOTRAIN_PROCESS = None
        return message, gr.update(visible=True), gr.update(visible=False)

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
