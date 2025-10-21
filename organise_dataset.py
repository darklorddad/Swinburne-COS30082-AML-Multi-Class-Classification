import os
import shutil
import re

def get_class_name_from_filename(filename):
    """
    Extracts class name from filename.
    e.g. 'Black_footed_Albatross_0004_2731401028.jpg' -> 'Black_footed_Albatross'
    """
    match = re.match(r'(.+?)_\d{3,}', filename)
    if match:
        return match.group(1)
    return None

def create_class_mapping(train_txt_path):
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
        print(f"Error: {train_txt_path} not found.")
        return None
    return class_mapping

def process_dataset(base_dir, annotations_file, source_subdir, dest_dir, class_mapping):
    """
    Moves files from source to destination based on annotations.
    """
    annotations_path = os.path.join(base_dir, annotations_file)
    source_dir = os.path.join(base_dir, source_subdir)

    if not os.path.exists(annotations_path):
        print(f"Warning: Annotations file not found: {annotations_path}")
        return

    if not os.path.exists(source_dir):
        print(f"Warning: Source directory not found: {source_dir}")
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
                        print(f"Moving {source_path} to {dest_path}")
                        shutil.move(source_path, dest_path)
                    else:
                        print(f"Warning: Source file not found: {source_path}")
                else:
                    print(f"Warning: Class ID {class_id} for file {filename} not in mapping.")

def main():
    """
    Main function to organise the dataset.
    """
    base_dataset_dir = 'Dataset'
    processed_dir = os.path.join(base_dataset_dir, 'Processed_Dataset')
    
    # Create the main processed directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create class mapping from train.txt
    train_txt_path = os.path.join(base_dataset_dir, 'train.txt')
    class_mapping = create_class_mapping(train_txt_path)
    
    if not class_mapping:
        print("Error: Could not create class mapping from train.txt. Aborting.")
        return
        
    print("Class mapping created:")
    for class_id, class_name in sorted(class_mapping.items()):
        print(f"  {class_id}: {class_name}")
        
    # Process training set
    print("\nProcessing training set...")
    process_dataset(base_dataset_dir, 'train.txt', 'Train', processed_dir, class_mapping)
    
    # Process test set
    print("\nProcessing test set...")
    process_dataset(base_dataset_dir, 'test.txt', 'Test', processed_dir, class_mapping)
    
    print("\nDataset organisation complete.")
    print(f"Files are now in subdirectories inside {processed_dir}")

if __name__ == '__main__':
    main()
