import tkinter as tk
from tkinter import filedialog, messagebox
import os

class ManifestGeneratorApp:
    """
    A simple GUI application to generate a manifest of files in a directory.
    """
    def __init__(self, root):
        """
        Initialise the application window and widgets.
        """
        self.root = root
        self.root.title("Directory Manifest Generator")
        self.directory_path = tk.StringVar()

        # Frame for directory selection
        frame = tk.Frame(root, padx=10, pady=10)
        frame.pack(padx=10, pady=10)

        # Directory selection button
        select_button = tk.Button(frame, text="Select Directory", command=self.select_directory)
        select_button.pack(side=tk.LEFT)

        # Label to display selected directory
        dir_label = tk.Label(frame, textvariable=self.directory_path, relief=tk.SUNKEN, width=50)
        dir_label.pack(side=tk.LEFT, padx=10)

        # Generate manifest button
        generate_button = tk.Button(root, text="Generate Manifest", command=self.generate_manifest)
        generate_button.pack(pady=5)

    def select_directory(self):
        """
        Open a dialog to select a directory and update the path variable.
        """
        directory = filedialog.askdirectory()
        if directory:
            self.directory_path.set(directory)

    def generate_manifest(self):
        """
        Generate a manifest.txt file for the selected directory.
        """
        directory = self.directory_path.get()
        if not directory:
            messagebox.showerror("Error", "Please select a directory first.")
            return

        manifest_path = os.path.join(directory, "manifest.txt")
        
        # Common directories and files to ignore
        ignored_dirs = {'.git', '__pycache__', '.vscode', '.idea', 'node_modules', 'venv', '.venv'}
        ignored_files = {'manifest.txt'}
        # Ignoring binary/non-text files by extension is a good idea for prompt context
        ignored_extensions = {
            # Compiled
            '.pyc', '.pyo', '.pyd', '.o', '.so', '.dll', '.exe',
            # Archives
            '.zip', '.tar', '.gz', '.rar', '.7z',
            # Images
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
            # Documents
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            # Other
            '.log', '.tmp', '.bak', '.swp'
        }

        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                for root, dirs, files in os.walk(directory, topdown=True):
                    # Exclude ignored directories from traversal
                    dirs[:] = [d for d in dirs if d not in ignored_dirs]

                    for filename in files:
                        if filename in ignored_files:
                            continue
                        
                        _, extension = os.path.splitext(filename)
                        if extension.lower() in ignored_extensions:
                            continue

                        full_path = os.path.join(root, filename)
                        # Use forward slashes for consistency in manifest
                        relative_path = os.path.relpath(full_path, directory).replace(os.sep, '/')
                        f.write(f"{relative_path}\n")

            messagebox.showinfo("Success", f"Manifest file 'manifest.txt' created in:\n{directory}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ManifestGeneratorApp(root)
    root.mainloop()
