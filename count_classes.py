import os
import tkinter as tk
from tkinter import filedialog, scrolledtext
import threading
from collections import Counter

class ClassCounterApp:
    """
    A simple GUI application to count classes and items per class in a dataset directory.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Class Counter")
        self.root.geometry("600x400")

        self.target_dir = tk.StringVar()

        # Frame for directory selection
        dir_frame = tk.Frame(self.root, padx=10, pady=10)
        dir_frame.pack(fill=tk.X)

        dir_label = tk.Label(dir_frame, text="Dataset Directory:")
        dir_label.pack(side=tk.LEFT)

        self.dir_entry = tk.Entry(dir_frame, textvariable=self.target_dir, width=50)
        self.dir_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        browse_button = tk.Button(dir_frame, text="Browse...", command=self.select_directory)
        browse_button.pack(side=tk.LEFT)

        # Frame for controls
        control_frame = tk.Frame(self.root, padx=10, pady=5)
        control_frame.pack(fill=tk.X)

        run_button = tk.Button(control_frame, text="Count Classes", command=self.start_counting)
        run_button.pack(side=tk.LEFT)

        # Frame for logging
        log_frame = tk.Frame(self.root, padx=10, pady=10)
        log_frame.pack(expand=True, fill=tk.BOTH)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(expand=True, fill=tk.BOTH)

    def log(self, message):
        """Append a message to the log text widget."""
        self.root.after(0, self._log_thread_safe, message)

    def _log_thread_safe(self, message):
        """Thread-safe method to append a message to the log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def select_directory(self):
        """Open a dialog to select a directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.target_dir.set(directory)
            self.log(f"Selected directory: {directory}")

    def start_counting(self):
        """Start the class counting process in a new thread."""
        target_dir = self.target_dir.get()
        if not target_dir:
            self.log("Please select a directory first.")
            return

        self.log("Starting class count...")
        thread = threading.Thread(target=self.run_counting_logic, args=(target_dir,))
        thread.daemon = True
        thread.start()

    def run_counting_logic(self, target_dir):
        """The logic for counting classes and files."""
        if not os.path.isdir(target_dir):
            self.log(f"Error: Directory not found at '{target_dir}'")
            return

        try:
            entries = os.listdir(target_dir)
            class_dirs = [entry for entry in entries if os.path.isdir(os.path.join(target_dir, entry))]

            if not class_dirs:
                self.log(f"No class subdirectories found in '{target_dir}'.")
                return

            self.log(f"Found {len(class_dirs)} classes.")
            self.log("-" * 20)

            class_counts = Counter()
            for class_name in sorted(class_dirs):
                class_path = os.path.join(target_dir, class_name)
                num_files = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
                class_counts[class_name] = num_files
            
            for class_name, count in class_counts.items():
                self.log(f"{class_name}: {count} items")
            
            self.log("-" * 20)
            self.log("Counting complete.")

        except OSError as e:
            self.log(f"Error accessing directory '{target_dir}': {e}")

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = ClassCounterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
