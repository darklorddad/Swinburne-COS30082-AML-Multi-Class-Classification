import json
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_metrics(json_path):
    """
    Loads trainer_state.json and plots training and evaluation metrics.

    Args:
        json_path (str): The path to the trainer_state.json file.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file was not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file at {json_path}")
        return

    log_history = data.get('log_history')
    if not log_history:
        print("No 'log_history' found in the JSON file.")
        return

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
    plt.show()

if __name__ == '__main__':
    # Path to the JSON file provided by the user
    file_path = r'Model-Swin-Transformer-88\checkpoint-1275\trainer_state.json'
    plot_training_metrics(file_path)
