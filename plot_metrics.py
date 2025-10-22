import json
import os
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

def plot_training_metrics(json_path):
    """
    Loads trainer_state.json and creates plots for training and evaluation metrics.

    Args:
        json_path (str): The path to the trainer_state.json file.

    Returns:
        dict: A dictionary of matplotlib figures, with chart titles as keys.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file was not found at {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Could not decode JSON from the file at {json_path}")

    log_history = data.get('log_history')
    if not log_history:
        raise ValueError("No 'log_history' found in the JSON file.")

    df = pd.DataFrame(log_history)

    # Separate training and evaluation logs
    train_df = df[df['loss'].notna()].copy()
    eval_df = df[df['eval_loss'].notna()].copy()

    # Convert step to numeric, just in case
    if 'step' in train_df.columns:
        train_df['step'] = pd.to_numeric(train_df['step'])
    if 'step' in eval_df.columns:
        eval_df['step'] = pd.to_numeric(eval_df['step'])

    figures = {}

    # Plot 1: Loss (Training and Evaluation)
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    if not train_df.empty and 'step' in train_df.columns and 'loss' in train_df.columns:
        ax_loss.plot(train_df['step'], train_df['loss'], label='Training Loss', marker='o', linestyle='-')
    if not eval_df.empty and 'step' in eval_df.columns and 'eval_loss' in eval_df.columns:
        ax_loss.plot(eval_df['step'], eval_df['eval_loss'], label='Evaluation Loss', marker='x', linestyle='--')
    ax_loss.set_xlabel('Step')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training vs. Evaluation Loss')
    ax_loss.legend()
    ax_loss.grid(True)
    figures['Loss'] = fig_loss

    # Plot 2: Evaluation Accuracy
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    if not eval_df.empty and 'step' in eval_df.columns and 'eval_accuracy' in eval_df.columns:
        ax_acc.plot(eval_df['step'], eval_df['eval_accuracy'], label='Evaluation Accuracy', marker='o', linestyle='-', color='g')
    ax_acc.set_xlabel('Step')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Evaluation Accuracy')
    ax_acc.legend()
    ax_acc.grid(True)
    figures['Accuracy'] = fig_acc

    # Plot 3: Learning Rate
    fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
    if not train_df.empty and 'step' in train_df.columns and 'learning_rate' in train_df.columns:
        ax_lr.plot(train_df['step'], train_df['learning_rate'], label='Learning Rate', marker='o', linestyle='-', color='r')
    ax_lr.set_xlabel('Step')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.set_title('Learning Rate Schedule')
    ax_lr.legend()
    ax_lr.grid(True)
    figures['Learning Rate'] = fig_lr

    # Plot 4: Grad Norm
    fig_gn, ax_gn = plt.subplots(figsize=(10, 6))
    if not train_df.empty and 'step' in train_df.columns and 'grad_norm' in train_df.columns:
        ax_gn.plot(train_df['step'], train_df['grad_norm'], label='Grad Norm', marker='o', linestyle='-', color='purple')
    ax_gn.set_xlabel('Step')
    ax_gn.set_ylabel('Grad Norm')
    ax_gn.set_title('Gradient Norm')
    ax_gn.legend()
    ax_gn.grid(True)
    figures['Gradient Norm'] = fig_gn

    # Plot 5: Evaluation F1 Scores
    fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))
    if not eval_df.empty and 'step' in eval_df.columns:
        if 'eval_f1_macro' in eval_df.columns:
            ax_f1.plot(eval_df['step'], eval_df['eval_f1_macro'], label='F1 Macro', marker='o', linestyle='-')
        if 'eval_f1_micro' in eval_df.columns:
            ax_f1.plot(eval_df['step'], eval_df['eval_f1_micro'], label='F1 Micro', marker='x', linestyle='--')
        if 'eval_f1_weighted' in eval_df.columns:
            ax_f1.plot(eval_df['step'], eval_df['eval_f1_weighted'], label='F1 Weighted', marker='s', linestyle=':')
    ax_f1.set_xlabel('Step')
    ax_f1.set_ylabel('F1 Score')
    ax_f1.set_title('Evaluation F1 Scores')
    ax_f1.legend()
    ax_f1.grid(True)
    figures['F1 Scores'] = fig_f1

    # Plot 6: Evaluation Precision Scores
    fig_prec, ax_prec = plt.subplots(figsize=(10, 6))
    if not eval_df.empty and 'step' in eval_df.columns:
        if 'eval_precision_macro' in eval_df.columns:
            ax_prec.plot(eval_df['step'], eval_df['eval_precision_macro'], label='Precision Macro', marker='o', linestyle='-')
        if 'eval_precision_micro' in eval_df.columns:
            ax_prec.plot(eval_df['step'], eval_df['eval_precision_micro'], label='Precision Micro', marker='x', linestyle='--')
        if 'eval_precision_weighted' in eval_df.columns:
            ax_prec.plot(eval_df['step'], eval_df['eval_precision_weighted'], label='Precision Weighted', marker='s', linestyle=':')
    ax_prec.set_xlabel('Step')
    ax_prec.set_ylabel('Precision Score')
    ax_prec.set_title('Evaluation Precision Scores')
    ax_prec.legend()
    ax_prec.grid(True)
    figures['Precision'] = fig_prec

    # Plot 7: Evaluation Recall Scores
    fig_recall, ax_recall = plt.subplots(figsize=(10, 6))
    if not eval_df.empty and 'step' in eval_df.columns:
        if 'eval_recall_macro' in eval_df.columns:
            ax_recall.plot(eval_df['step'], eval_df['eval_recall_macro'], label='Recall Macro', marker='o', linestyle='-')
        if 'eval_recall_micro' in eval_df.columns:
            ax_recall.plot(eval_df['step'], eval_df['eval_recall_micro'], label='Recall Micro', marker='x', linestyle='--')
        if 'eval_recall_weighted' in eval_df.columns:
            ax_recall.plot(eval_df['step'], eval_df['eval_recall_weighted'], label='Recall Weighted', marker='s', linestyle=':')
    ax_recall.set_xlabel('Step')
    ax_recall.set_ylabel('Recall Score')
    ax_recall.set_title('Evaluation Recall Scores')
    ax_recall.legend()
    ax_recall.grid(True)
    figures['Recall'] = fig_recall

    # Plot 8: Epoch Progression
    fig_epoch, ax_epoch = plt.subplots(figsize=(10, 6))
    if not df.empty and 'step' in df.columns and 'epoch' in df.columns:
        epoch_df = df[['step', 'epoch']].dropna().drop_duplicates('step').sort_values('step')
        ax_epoch.plot(epoch_df['step'], epoch_df['epoch'], label='Epoch', marker='.', linestyle='-')
    ax_epoch.set_xlabel('Step')
    ax_epoch.set_ylabel('Epoch')
    ax_epoch.set_title('Epoch Progression')
    ax_epoch.legend()
    ax_epoch.grid(True)
    figures['Epoch'] = fig_epoch

    # Plot 9: Evaluation Runtime
    fig_runtime, ax_runtime = plt.subplots(figsize=(10, 6))
    if not eval_df.empty and 'step' in eval_df.columns and 'eval_runtime' in eval_df.columns:
        ax_runtime.plot(eval_df['step'], eval_df['eval_runtime'], label='Eval Runtime', marker='o', linestyle='-')
    ax_runtime.set_xlabel('Step')
    ax_runtime.set_ylabel('Runtime (s)')
    ax_runtime.set_title('Evaluation Runtime')
    ax_runtime.legend()
    ax_runtime.grid(True)
    figures['Eval Runtime'] = fig_runtime

    # Plot 10: Evaluation Samples Per Second
    fig_sps, ax_sps = plt.subplots(figsize=(10, 6))
    if not eval_df.empty and 'step' in eval_df.columns and 'eval_samples_per_second' in eval_df.columns:
        ax_sps.plot(eval_df['step'], eval_df['eval_samples_per_second'], label='Eval Samples/sec', marker='o', linestyle='-')
    ax_sps.set_xlabel('Step')
    ax_sps.set_ylabel('Samples/sec')
    ax_sps.set_title('Evaluation Samples Per Second')
    ax_sps.legend()
    ax_sps.grid(True)
    figures['Eval Samples/sec'] = fig_sps

    # Plot 11: Evaluation Steps Per Second
    fig_steps_ps, ax_steps_ps = plt.subplots(figsize=(10, 6))
    if not eval_df.empty and 'step' in eval_df.columns and 'eval_steps_per_second' in eval_df.columns:
        ax_steps_ps.plot(eval_df['step'], eval_df['eval_steps_per_second'], label='Eval Steps/sec', marker='o', linestyle='-')
    ax_steps_ps.set_xlabel('Step')
    ax_steps_ps.set_ylabel('Steps/sec')
    ax_steps_ps.set_title('Evaluation Steps Per Second')
    ax_steps_ps.legend()
    ax_steps_ps.grid(True)
    figures['Eval Steps/sec'] = fig_steps_ps

    return figures

if __name__ == '__main__':

    def select_file(entry):
        file_path = filedialog.askopenfilename(
            title="Select trainer_state.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def select_dir(entry):
        dir_path = filedialog.askdirectory(title="Select Save Directory")
        if dir_path:
            entry.delete(0, tk.END)
            entry.insert(0, dir_path)

    def display_charts_in_tabs(figures, parent):
        chart_window = tk.Toplevel(parent)
        chart_window.title("Training & Evaluation Metrics")
        chart_window.geometry("800x600")

        notebook = ttk.Notebook(chart_window)
        notebook.pack(expand=True, fill='both', padx=5, pady=5)

        for title, fig in figures.items():
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=title)

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            plt.close(fig)  # Close the figure to free up memory

    def generate_charts():
        json_path = json_path_entry.get()
        save_dir = save_dir_entry.get()
        if not json_path:
            status_label.config(text="Please select a JSON file.", fg="red")
            return

        try:
            figures = plot_training_metrics(json_path)

            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                for title, fig in figures.items():
                    filename = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
                    save_path = os.path.join(save_dir, f"{filename}.png")
                    fig.savefig(save_path)
                    plt.close(fig)
                status_label.config(text=f"Charts saved to {save_dir}", fg="green")
            else:
                display_charts_in_tabs(figures, root)
                status_label.config(text="Displaying charts in new window.", fg="green")

        except Exception as e:
            status_label.config(text=f"Error: {e}", fg="red")

    root = tk.Tk()
    root.title("Training Metrics Plotter")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack()

    # JSON file selection
    json_path_label = tk.Label(frame, text="trainer_state.json path:")
    json_path_label.grid(row=0, column=0, sticky="w", pady=2)

    json_path_entry = tk.Entry(frame, width=50)
    json_path_entry.grid(row=0, column=1, pady=2)

    json_browse_button = tk.Button(frame, text="Browse...", command=lambda: select_file(json_path_entry))
    json_browse_button.grid(row=0, column=2, padx=5, pady=2)

    # Save directory selection
    save_dir_label = tk.Label(frame, text="Save charts to (optional):")
    save_dir_label.grid(row=1, column=0, sticky="w", pady=2)

    save_dir_entry = tk.Entry(frame, width=50)
    save_dir_entry.grid(row=1, column=1, pady=2)

    save_dir_browse_button = tk.Button(frame, text="Browse...", command=lambda: select_dir(save_dir_entry))
    save_dir_browse_button.grid(row=1, column=2, padx=5, pady=2)

    # Generate button
    generate_button = tk.Button(frame, text="Generate Charts", command=generate_charts)
    generate_button.grid(row=2, column=1, pady=10)

    # Status label
    status_label = tk.Label(frame, text="", fg="green")
    status_label.grid(row=3, column=0, columnspan=3, pady=2)

    root.mainloop()
