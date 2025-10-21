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

    manifest = {
        "included_classes": {},
        "skipped_classes": {},
    }
    processed_class_names = set()

    required_images = min_images_per_split * 2  # train, validation

    for class_dir in class_dirs:
        base_class_name = os.path.basename(class_dir)
        class_name = base_class_name
        counter = 1
        while class_name in processed_class_names:
            class_name = f"{base_class_name}_{counter}"
            counter += 1
        processed_class_names.add(class_name)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        images = [
            f
            for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f)) and os.path.splitext(f)[1].lower() in image_extensions
        ]

        if len(images) < required_images:
            manifest["skipped_classes"][class_name] = {
                "original_path": class_dir,
                "count": len(images),
                "reason": f"Not enough images. Found {len(images)}, required {required_images}.",
            }
            continue

        random.shuffle(images)

        # Determine the number of validation images
        num_validation = int(len(images) * 0.20)
        if num_validation < min_images_per_split:
            num_validation = min_images_per_split

        # Ensure there's at least min_images_per_split for training
        if len(images) - num_validation < min_images_per_split:
            manifest["skipped_classes"][class_name] = {
                "original_path": class_dir,
                "count": len(images),
                "reason": f"Not enough images for a train/validation split. Found {len(images)}, required at least {num_validation + min_images_per_split}.",
            }
            continue

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
