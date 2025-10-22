import os
from collections import Counter
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import matplotlib.pyplot as plt
import numpy as np

def get_class_from_line(line: str) -> str:
    """Extracts the class name from a line in the manifest file."""
    # Example line: - acadian_flycatcher/acadian_flycatcher_0001.jpg
    # The class is the part before the first '/'
    if '/' in line:
        # Strip leading '- ' and then split
        return line.strip().lstrip('- ').split('/')[0]
    return None

class DatasetBalanceApp:
    """
    A simple GUI application to analyse the balance of a dataset from a manifest file.
    """
    def __init__(self, root):
        """
        Initialise the application.
        """
        self.root = root
        self.root.title("Dataset Balance Checker")

        self.frame = tk.Frame(root, padx=10, pady=10)
        self.frame.pack(padx=10, pady=10)

        # Manifest file selection
        self.manifest_label = tk.Label(self.frame, text="Manifest File:")
        self.manifest_label.grid(row=0, column=0, sticky=tk.W, pady=2)

        self.manifest_path_var = tk.StringVar()
        self.manifest_entry = tk.Entry(self.frame, textvariable=self.manifest_path_var, width=50)
        self.manifest_entry.grid(row=0, column=1, pady=2)

        self.browse_button = tk.Button(self.frame, text="Browse...", command=self.select_manifest_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=2)

        # Run button
        self.run_button = tk.Button(self.frame, text="Analyse Balance", command=self.run_analysis)
        self.run_button.grid(row=1, column=0, columnspan=3, pady=10)

        # Log area
        self.log_area = scrolledtext.ScrolledText(self.frame, width=80, height=20)
        self.log_area.grid(row=2, column=0, columnspan=3, pady=5)

    def log(self, message):
        """
        Log a message to the text area.
        """
        self.log_area.insert(tk.END, message + '\n')
        self.log_area.see(tk.END)

    def select_manifest_file(self):
        """
        Open a file dialog to select the manifest file.
        """
        filepath = filedialog.askopenfilename(
            title="Select Manifest File",
            filetypes=(("Markdown files", "*.md"), ("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filepath:
            self.manifest_path_var.set(filepath)

    def run_analysis(self):
        """
        Run the dataset balance analysis.
        """
        manifest_path = self.manifest_path_var.get()
        if not manifest_path:
            messagebox.showerror("Error", "Please select a manifest file.")
            return

        self.log_area.delete('1.0', tk.END)
        self.log(f"Starting analysis for: {manifest_path}")

        if not os.path.exists(manifest_path):
            self.log(f"Error: Manifest file not found at '{manifest_path}'")
            messagebox.showerror("Error", f"Manifest file not found at '{manifest_path}'")
            return

        with open(manifest_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        class_names = [get_class_from_line(line) for line in lines if get_class_from_line(line)]
        class_counts = Counter(class_names)

        if not class_counts:
            self.log("No classes found in the manifest file.")
            messagebox.showinfo("Info", "No classes found in the manifest file.")
            return

        counts = list(class_counts.values())
        min_count = min(counts)
        max_count = max(counts)
        total_images = sum(counts)
        num_classes = len(class_counts)
        mean_count = np.mean(counts)
        std_dev = np.std(counts)
        imbalance_ratio = max_count / min_count

        self.log("Dataset Balance Analysis")
        self.log("="*25)
        self.log(f"Manifest file: {manifest_path}")
        self.log(f"Total classes: {num_classes}")
        self.log(f"Total images: {total_images}")
        self.log("\nImages per class:")
        self.log(f"  - Minimum: {min_count}")
        self.log(f"  - Maximum: {max_count}")
        self.log(f"  - Average: {mean_count:.2f}")
        self.log(f"  - Std Dev: {std_dev:.2f}")
        self.log(f"\nImbalance Ratio (Max/Min): {imbalance_ratio:.2f}:1")

        # Plotting the distribution
        sorted_classes = sorted(class_counts.keys())
        sorted_counts = [class_counts[c] for c in sorted_classes]

        plt.figure(figsize=(20, 10))
        plt.bar(sorted_classes, sorted_counts)
        plt.xlabel('Class')
        plt.ylabel('Number of Images')
        plt.title('Image Distribution Across Classes')
        plt.xticks(rotation=90, fontsize='small')
        plt.tight_layout()
        
        plot_filename = 'class_distribution.png'
        plt.savefig(plot_filename)
        self.log(f"\nPlot of class distribution saved to '{plot_filename}'")
        messagebox.showinfo("Success", f"Analysis complete. Plot saved to {plot_filename}")

def main():
    """Main function to run the script."""
    root = tk.Tk()
    app = DatasetBalanceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
