import os
import shutil
import re
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading

def get_class_name_from_filename(filename):
    """
    Extracts class name from filename.
    e.g. 'Black_footed_Albatross_0004_2731401028.jpg' -> 'Black_footed_Albatross'
    """
    match = re.match(r'(.+?)_\d{3,}', filename)
    if match:
        return match.group(1)
    return None

def create_class_mapping(train_txt_path, logger=print):
    """
    Creates a mapping from class ID to class name from train.txt.
    """
    class_mapping = {}
    try:
        with open(train_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, class_id_str = parts
                    class_id = int(class_id_str)
                    if class_id not in class_mapping:
                        class_name = get_class_name_from_filename(filename)
                        if class_name:
                            class_mapping[class_id] = class_name
    except FileNotFoundError:
        logger(f"Error: {train_txt_path} not found.")
        return None
    return class_mapping

def process_dataset(base_dir, annotations_file, source_subdir, dest_dir, class_mapping, logger=print):
    """
    Moves files from source to destination based on annotations.
    """
    annotations_path = os.path.join(base_dir, annotations_file)
    source_dir = os.path.join(base_dir, source_subdir)

    if not os.path.exists(annotations_path):
        logger(f"Warning: Annotations file not found: {annotations_path}")
        return

    if not os.path.exists(source_dir):
        logger(f"Warning: Source directory not found: {source_dir}")
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
                        logger(f"Moving {source_path} to {dest_path}")
                        shutil.move(source_path, dest_path)
                    else:
                        logger(f"Warning: Source file not found: {source_path}")
                else:
                    logger(f"Warning: Class ID {class_id} for file {filename} not in mapping.")

class OrganiseDatasetApp:
    """
    A simple GUI application to organise the dataset.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Organiser")
        self.root.geometry("800x600")

        self.log_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled')
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.organise_button = tk.Button(self.root, text="Organise Dataset", command=self.start_organisation)
        self.organise_button.pack(pady=10)

    def log(self, message):
        """Appends a message to the log text widget."""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_organisation(self):
        """Starts the organisation process in a new thread to keep the GUI responsive."""
        self.organise_button.config(state=tk.DISABLED, text="Organising...")
        thread = threading.Thread(target=self.run_organisation_logic)
        thread.daemon = True
        thread.start()

    def run_organisation_logic(self):
        """
        The main logic for organising the dataset.
        """
        base_dataset_dir = 'Dataset'
        processed_dir = os.path.join(base_dataset_dir, 'Processed_Dataset')
        
        # Create the main processed directory
        os.makedirs(processed_dir, exist_ok=True)
        
        # Create class mapping from train.txt
        train_txt_path = os.path.join(base_dataset_dir, 'train.txt')
        class_mapping = create_class_mapping(train_txt_path, logger=self.log)
        
        if not class_mapping:
            self.log("Error: Could not create class mapping from train.txt. Aborting.")
            self.organise_button.config(state=tk.NORMAL, text="Organise Dataset")
            messagebox.showerror("Error", "Could not create class mapping from train.txt. Aborting.")
            return
            
        self.log("Class mapping created:")
        for class_id, class_name in sorted(class_mapping.items()):
            self.log(f"  {class_id}: {class_name}")
            
        # Process training set
        self.log("\nProcessing training set...")
        process_dataset(base_dataset_dir, 'train.txt', 'Train', processed_dir, class_mapping, logger=self.log)
        
        # Process test set
        self.log("\nProcessing test set...")
        process_dataset(base_dataset_dir, 'test.txt', 'Test', processed_dir, class_mapping, logger=self.log)
        
        self.log("\nDataset organisation complete.")
        self.log(f"Files are now in subdirectories inside {processed_dir}")
        self.organise_button.config(state=tk.NORMAL, text="Organise Dataset")
        messagebox.showinfo("Complete", f"Dataset organisation complete.\nFiles are in {processed_dir}")

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = OrganiseDatasetApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
