import os
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading

class ImageFilenameProcessorApp:
    """
    A simple GUI application to process image filenames in a target directory.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filename Processor")
        self.root.geometry("800x600")

        # --- Options ---
        self.to_lowercase = tk.BooleanVar(value=True)
        self.to_standardise = tk.BooleanVar(value=False)

        # Directory selection frame
        dir_frame = tk.Frame(self.root)
        dir_frame.pack(padx=10, pady=5, fill=tk.X)

        dir_label = tk.Label(dir_frame, text="Target Directory:")
        dir_label.pack(side=tk.LEFT, padx=(0, 5))

        self.dir_entry = tk.Entry(dir_frame)
        self.dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_button = tk.Button(dir_frame, text="Browse...", command=self.select_directory)
        browse_button.pack(side=tk.LEFT, padx=(5, 0))

        # Options frame
        options_frame = tk.Frame(self.root)
        options_frame.pack(padx=10, pady=5, fill=tk.X, anchor=tk.W)

        lowercase_check = tk.Checkbutton(options_frame, text="Convert filenames to lowercase", variable=self.to_lowercase)
        lowercase_check.pack(anchor=tk.W)

        standardise_check = tk.Checkbutton(options_frame, text="Standardise filenames (e.g., class_name_0001.jpg)", variable=self.to_standardise)
        standardise_check.pack(anchor=tk.W)

        # Log text area
        self.log_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled')
        self.log_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Action button
        self.process_button = tk.Button(self.root, text="Process Image Names", command=self.start_processing)
        self.process_button.pack(pady=10)

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

    def start_processing(self):
        """Starts the processing in a new thread."""
        target_dir = self.dir_entry.get()
        if not target_dir or not os.path.isdir(target_dir):
            messagebox.showerror("Error", "Please select a valid directory.")
            return

        to_lowercase_val = self.to_lowercase.get()
        to_standardise_val = self.to_standardise.get()

        if not to_lowercase_val and not to_standardise_val:
            messagebox.showinfo("Info", "No action selected. Please check at least one box.")
            return

        self.process_button.config(state=tk.DISABLED, text="Processing...")
        thread = threading.Thread(target=self.run_processing_logic, args=(target_dir, to_lowercase_val, to_standardise_val))
        thread.daemon = True
        thread.start()

    def run_processing_logic(self, target_dir, to_lowercase, to_standardise):
        """
        The core logic for processing image filenames recursively.
        """
        self.log(f"\nStarting image name processing in '{target_dir}'...")
        
        try:
            if to_standardise:
                self.log("Standardising filenames...")
                self.standardise_filenames(target_dir, to_lowercase)
            elif to_lowercase:
                self.log("Converting filenames to lowercase...")
                self.lowercase_filenames(target_dir)

            self.log("\nProcessing complete.")
            self.root.after(0, lambda: messagebox.showinfo("Complete", "Processing has finished."))

        except Exception as e:
            self.log(f"An unexpected error occurred: {e}")
            self.root.after(0, lambda e=e: messagebox.showerror("Error", f"An unexpected error occurred: {e}"))
        finally:
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL, text="Process Image Names"))

    def lowercase_filenames(self, target_dir):
        """
        Recursively converts filenames to lowercase.
        """
        file_count = 0
        renamed_count = 0
        for dirpath, _, filenames in os.walk(target_dir):
            for old_name in filenames:
                file_count += 1
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

                    os.rename(old_path, temp_path)
                    os.rename(temp_path, new_path)
                    self.log(f"Renamed '{os.path.basename(old_path)}' to '{os.path.basename(new_path)}' in '{os.path.basename(dirpath)}'")
                    renamed_count += 1
                except OSError as e:
                    self.log(f"Error renaming '{old_path}': {e}")

        if file_count == 0:
            self.log("No files found to process.")
        else:
            self.log(f"\nProcessed {file_count} files, renamed {renamed_count}.")

    def standardise_filenames(self, target_dir, to_lowercase):
        """
        Standardises filenames to {class_name}_{counter}.{ext}.
        """
        total_renamed = 0
        for dirpath, _, filenames in os.walk(target_dir):
            if not filenames:
                continue

            class_name = os.path.basename(dirpath)
            if to_lowercase:
                class_name = class_name.lower()
            
            self.log(f"\nProcessing directory: {class_name}")
            
            files_to_rename = sorted(filenames)
            
            rename_map = []
            counter = 1
            for old_name in files_to_rename:
                _, extension = os.path.splitext(old_name)
                if to_lowercase:
                    extension = extension.lower()
                
                new_name = f"{class_name}_{counter:04d}{extension}"
                
                old_path = os.path.join(dirpath, old_name)
                new_path = os.path.join(dirpath, new_name)
                
                rename_map.append({'old_path': old_path, 'new_path': new_path})
                counter += 1

            # First pass: rename to temporary names to avoid collisions
            for item in rename_map:
                temp_path = item['old_path'] + '.tmp'
                os.rename(item['old_path'], temp_path)
                item['temp_path'] = temp_path

            # Second pass: rename from temporary to final names
            for item in rename_map:
                os.rename(item['temp_path'], item['new_path'])
                self.log(f"Renamed '{os.path.basename(item['old_path'])}' to '{os.path.basename(item['new_path'])}'")
                total_renamed += 1
        
        self.log(f"\nStandardised {total_renamed} files.")

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = ImageFilenameProcessorApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
