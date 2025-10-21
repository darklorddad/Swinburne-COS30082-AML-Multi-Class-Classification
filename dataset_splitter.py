import argparse
import json
import os
import random
import shutil


def split_image_dataset(source_dir: str, output_dir: str, min_images_per_split: int = 5):
    """
    Splits an image dataset into training and validation sets, creating separate directories for each.

    The function identifies class directories as the deepest directories in the source
    path that contain files. It then splits the images in each class into train and
    validation sets, ensuring each set has a minimum number of images.
    Classes with insufficient images are skipped.

    A manifest file is created to document the split, including which classes were
    included or skipped, and the counts of images in each split.

    Args:
        source_dir (str): The path to the source dataset directory.
        output_dir (str): The base path for the output directories. This will be used as a prefix,
                          creating `{output_dir}_train` and `{output_dir}_validation`.
        min_images_per_split (int): The minimum number of images required for each
                                    of the training and validation sets for a class to be included.
    """
    train_dir = f"{output_dir}_train"
    validation_dir = f"{output_dir}_validation"

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)

    class_dirs = []
    for root, dirs, files in os.walk(source_dir):
        # A directory is a class directory if it has no subdirectories but contains files.
        if not dirs and files:
            class_dirs.append(root)

    # Pre-scan to validate dataset requirements
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    required_total_images_per_class = min_images_per_split * 2
    valid_class_count = 0
    class_image_data = {}

    for class_dir in class_dirs:
        images = [
            f
            for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f)) and os.path.splitext(f)[1].lower() in image_extensions
        ]
        class_image_data[class_dir] = images
        if len(images) >= required_total_images_per_class:
            valid_class_count += 1

    if valid_class_count < 2:
        print(f"Error: Dataset splitting requires at least 2 classes with a minimum of {required_total_images_per_class} images each.")
        print(f"Found {valid_class_count} valid classes.")
        return

    manifest = {
        "included_classes": {},
        "skipped_classes": {},
    }
    processed_class_names = set()

    for class_dir in class_dirs:
        base_class_name = os.path.basename(class_dir)
        class_name = base_class_name
        counter = 1
        while class_name in processed_class_names:
            class_name = f"{base_class_name}_{counter}"
            counter += 1
        processed_class_names.add(class_name)

        images = class_image_data[class_dir]
        required_total_images = min_images_per_split * 2

        if len(images) < required_total_images:
            manifest["skipped_classes"][class_name] = {
                "original_path": class_dir,
                "count": len(images),
                "reason": f"Not enough images. Found {len(images)}, required a total of {required_total_images}.",
            }
            continue

        random.shuffle(images)

        # Determine split using the two-tiered logic
        num_val_ratio = round(len(images) * 0.2)
        num_train_ratio = len(images) - num_val_ratio

        if num_val_ratio >= min_images_per_split and num_train_ratio >= min_images_per_split:
            # Use the 80/20 ratio split
            num_validation = num_val_ratio
        else:
            # Fall back to the fixed minimum split
            num_validation = min_images_per_split

        validation_images = images[:num_validation]
        train_images = images[num_validation:]

        manifest["included_classes"][class_name] = {
            "original_path": class_dir,
            "train": len(train_images),
            "validation": len(validation_images),
            "total": len(images),
        }

        # Create directories and copy files
        train_class_dir = os.path.join(train_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        for image in train_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(train_class_dir, image))

        validation_class_dir = os.path.join(validation_dir, class_name)
        os.makedirs(validation_class_dir, exist_ok=True)
        for image in validation_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(validation_class_dir, image))

    manifest_path = f"{output_dir}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(
        f"Dataset split complete. Training data in {train_dir}, validation data in {validation_dir}. Manifest saved to {manifest_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split an image dataset into train and validation sets.")
    parser.add_argument("--source-dir", type=str, required=True, help="The path to the source dataset directory.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The base path for the output directories (e.g., 'my_dataset' will create 'my_dataset_train' and 'my_dataset_validation').",
    )
    parser.add_argument(
        "--min-images-per-split",
        type=int,
        default=5,
        help="The minimum number of images for each of the training and validation sets.",
    )
    args = parser.parse_args()
    split_image_dataset(args.source_dir, args.output_dir, args.min_images_per_split)
