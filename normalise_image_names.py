import os
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading

class NormaliseImageNamesApp:
    """
    A simple GUI application to normalise image filenames in a target directory.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Image Name Normaliser")
        self.root.geometry("800x600")

        # Directory selection frame
        dir_frame = tk.Frame(self.root)
        dir_frame.pack(padx=10, pady=5, fill=tk.X)

        dir_label = tk.Label(dir_frame, text="Target Directory:")
        dir_label.pack(side=tk.LEFT, padx=(0, 5))

        self.dir_entry = tk.Entry(dir_frame)
        self.dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_button = tk.Button(dir_frame, text="Browse...", command=self.select_directory)
        browse_button.pack(side=tk.LEFT, padx=(5, 0))

        # Log text area
        self.log_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled')
        self.log_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Action button
        self.normalise_button = tk.Button(self.root, text="Normalise Image Names", command=self.start_normalisation)
        self.normalise_button.pack(pady=10)

    def log(self, message):
        """Appends a message to the log text widget."""
        self.root.after(0, self._log_thread_safe, message)

    def _log_thread_safe(self, message):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)

    def select_directory(self):
        """Opens a dialog to select a directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)
            self.log(f"Selected directory: {directory}")

    def start_normalisation(self):
        """Starts the normalisation process in a new thread."""
        target_dir = self.dir_entry.get()
        if not target_dir or not os.path.isdir(target_dir):
            messagebox.showerror("Error", "Please select a valid directory.")
            return

        self.normalise_button.config(state=tk.DISABLED, text="Normalising...")
        thread = threading.Thread(target=self.run_normalisation_logic, args=(target_dir,))
        thread.daemon = True
        thread.start()

    def run_normalisation_logic(self, target_dir):
        """
        The core logic for normalising image filenames recursively.
        """
        self.log(f"\nStarting image name normalisation in '{target_dir}'...")
        
        try:
            file_count = 0
            renamed_count = 0
            for dirpath, _, filenames in os.walk(target_dir):
                for old_name in filenames:
                    file_count += 1
                    # Normalisation logic: convert to lowercase
                    new_name = old_name.lower()

                    if old_name == new_name:
                        continue

                    old_path = os.path.join(dirpath, old_name)
                    new_path = os.path.join(dirpath, new_name)

                    if os.path.exists(new_path) and not os.path.samefile(old_path, new_path):
                        self.log(f"Warning: Cannot rename '{old_path}' to '{new_path}' because a different file with that name already exists. Skipping.")
                        continue
                    
                    try:
                        # Use a two-step rename process for case-insensitive filesystems
                        temp_name = old_name + "_temp_rename"
                        temp_path = os.path.join(dirpath, temp_name)

                        if os.path.exists(temp_path):
                            self.log(f"Warning: Temporary path '{temp_path}' already exists. Skipping rename for '{old_name}'.")
                            continue

                        # Step 1: old -> temp
                        os.rename(old_path, temp_path)
                        
                        # Step 2: temp -> new
                        os.rename(temp_path, new_path)
                        self.log(f"Renamed '{os.path.basename(old_path)}' to '{os.path.basename(new_path)}' in '{os.path.basename(dirpath)}'")
                        renamed_count += 1
                    except OSError as e:
                        self.log(f"Error renaming '{old_path}': {e}")

            if file_count == 0:
                self.log("No files found to normalise.")
            else:
                self.log(f"\nProcessed {file_count} files, renamed {renamed_count}.")

            self.log("\nNormalisation complete.")
            self.root.after(0, lambda: messagebox.showinfo("Complete", "Normalisation process has finished."))

        except Exception as e:
            self.log(f"An unexpected error occurred: {e}")
            self.root.after(0, lambda e=e: messagebox.showerror("Error", f"An unexpected error occurred: {e}"))
        finally:
            self.root.after(0, lambda: self.normalise_button.config(state=tk.NORMAL, text="Normalise Image Names"))

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = NormaliseImageNamesApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
