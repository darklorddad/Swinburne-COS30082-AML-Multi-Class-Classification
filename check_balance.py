import argparse
import os
from collections import Counter
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

def analyze_dataset_balance(manifest_path: str):
    """
    Analyzes the balance of a dataset from a manifest file.

    Args:
        manifest_path (str): The path to the manifest file.
    """
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at '{manifest_path}'")
        return

    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    class_names = [get_class_from_line(line) for line in lines if get_class_from_line(line)]
    class_counts = Counter(class_names)

    if not class_counts:
        print("No classes found in the manifest file.")
        return

    counts = list(class_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    total_images = sum(counts)
    num_classes = len(class_counts)
    mean_count = np.mean(counts)
    std_dev = np.std(counts)
    imbalance_ratio = max_count / min_count

    print("Dataset Balance Analysis")
    print("="*25)
    print(f"Manifest file: {manifest_path}")
    print(f"Total classes: {num_classes}")
    print(f"Total images: {total_images}")
    print("\nImages per class:")
    print(f"  - Minimum: {min_count}")
    print(f"  - Maximum: {max_count}")
    print(f"  - Average: {mean_count:.2f}")
    print(f"  - Std Dev: {std_dev:.2f}")
    print(f"\nImbalance Ratio (Max/Min): {imbalance_ratio:.2f}:1")

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
    print(f"\nPlot of class distribution saved to '{plot_filename}'")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Analyse the balance of an image dataset from a manifest file.")
    parser.add_argument("manifest_file", help="Path to the manifest file.")
    args = parser.parse_args()

    analyze_dataset_balance(args.manifest_file)

if __name__ == "__main__":
    main()
