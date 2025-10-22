import json
import os
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import pandas as pd

def plot_training_metrics(json_path, save_dir=None):
    """
    Loads trainer_state.json and plots training and evaluation metrics.

    Args:
        json_path (str): The path to the trainer_state.json file.
        save_dir (str, optional): Directory to save plots. If None, plots are shown.
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

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Evaluation Metrics', fontsize=16)

    # Plot 1: Loss (Training and Evaluation)
    if not train_df.empty and 'step' in train_df.columns and 'loss' in train_df.columns:
        axs[0, 0].plot(train_df['step'], train_df['loss'], label='Training Loss', marker='o', linestyle='-')
    if not eval_df.empty and 'step' in eval_df.columns and 'eval_loss' in eval_df.columns:
        axs[0, 0].plot(eval_df['step'], eval_df['eval_loss'], label='Evaluation Loss', marker='x', linestyle='--')
    axs[0, 0].set_xlabel('Step')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Training vs. Evaluation Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: Evaluation Accuracy
    if not eval_df.empty and 'step' in eval_df.columns and 'eval_accuracy' in eval_df.columns:
        axs[0, 1].plot(eval_df['step'], eval_df['eval_accuracy'], label='Evaluation Accuracy', marker='o', linestyle='-', color='g')
    axs[0, 1].set_xlabel('Step')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_title('Evaluation Accuracy')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: Learning Rate
    if not train_df.empty and 'step' in train_df.columns and 'learning_rate' in train_df.columns:
        axs[1, 0].plot(train_df['step'], train_df['learning_rate'], label='Learning Rate', marker='o', linestyle='-', color='r')
    axs[1, 0].set_xlabel('Step')
    axs[1, 0].set_ylabel('Learning Rate')
    axs[1, 0].set_title('Learning Rate Schedule')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot 4: Evaluation F1 Scores
    if not eval_df.empty and 'step' in eval_df.columns:
        if 'eval_f1_macro' in eval_df.columns:
            axs[1, 1].plot(eval_df['step'], eval_df['eval_f1_macro'], label='F1 Macro', marker='o', linestyle='-')
        if 'eval_f1_micro' in eval_df.columns:
            axs[1, 1].plot(eval_df['step'], eval_df['eval_f1_micro'], label='F1 Micro', marker='x', linestyle='--')
        if 'eval_f1_weighted' in eval_df.columns:
            axs[1, 1].plot(eval_df['step'], eval_df['eval_f1_weighted'], label='F1 Weighted', marker='s', linestyle=':')
    axs[1, 1].set_xlabel('Step')
    axs[1, 1].set_ylabel('F1 Score')
    axs[1, 1].set_title('Evaluation F1 Scores')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "training_metrics.png")
        fig.savefig(save_path)
        plt.close(fig)
        return f"Chart saved to {save_path}"

    plt.show()
    return "Displaying chart."

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

    def generate_charts():
        json_path = json_path_entry.get()
        save_dir = save_dir_entry.get()
        if not json_path:
            status_label.config(text="Please select a JSON file.", fg="red")
            return

        if not save_dir:
            save_dir = None

        try:
            status = plot_training_metrics(json_path, save_dir)
            status_label.config(text=status, fg="green")
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
