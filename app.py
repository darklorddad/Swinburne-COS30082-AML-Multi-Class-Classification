import gradio as gr
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image


# --- Prediction Function ---
def classify_bird(model_path: str, input_image: Image.Image) -> dict:
    """
    Classifies the bird species from an input image.

    Args:
        model_path: Path to the fine-tuned model directory.
        input_image: A PIL Image of a bird.

    Returns:
        A dictionary with labels as keys and their corresponding confidence
        scores as values.
    """
    # --- Load Model and Processor ---
    try:
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)
    except Exception as e:
        # Provide a helpful error message if the model files cannot be loaded.
        raise OSError(
            f"Error loading model from {model_path}. "
            f"Please ensure that '{model_path}' contains the necessary model files "
            f"('config.json', 'model.safetensors', 'preprocessor_config.json'). "
            f"Original error: {e}"
        )

    # Preprocess the input image using the image processor.
    # This converts the PIL image into a tensor that the model can understand.
    inputs = image_processor(images=input_image, return_tensors="pt")

    # Perform inference with the model.
    # We use torch.no_grad() to disable gradient calculations, which saves memory
    # and speeds up computation, as we are only doing inference.
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the raw output scores (logits) from the model.
    logits = outputs.logits

    # Convert logits to probabilities using the softmax function.
    # The probabilities for all classes will sum to 1.
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Get the top 5 predictions.
    # torch.topk returns the k largest values and their indices.
    top5_prob, top5_indices = torch.topk(probabilities, 5)

    # Create a dictionary of the top 5 predicted labels and their confidences.
    # The model's configuration (config.id2label) is used to map class indices
    # back to their human-readable string labels.
    confidences = {
        model.config.id2label[i.item()]: p.item() for i, p in zip(top5_indices, top5_prob)
    }
    return confidences

# --- Gradio Interface ---
# Create the Gradio web application interface.

# Define the input components.
inputs = [
    gr.Textbox(
        label="Model Path",
        value="./",
        info="Path to the directory containing model files.",
    ),
    gr.Image(type="pil", label="Upload a bird image"),
]

# Define the output component: a label that shows top predictions.
# 'num_top_classes=5' configures it to display the top 5 results nicely.
outputs = gr.Label(num_top_classes=5, label="Predictions")

# Define the title and description for the web app.
# This information is based on the assignment details.
title = "Bird Species Classification"
description = """
This application classifies bird species from an image. It is based on a model
fine-tuned on the Caltech-UCSD Birds 200 (CUB-200) dataset.
Upload an image of a bird to see the model's prediction.
"""

# Assemble the interface.
demo = gr.Interface(
    fn=classify_bird,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
)

# --- Launch the App ---
if __name__ == "__main__":
    # Launch the Gradio app. It will be accessible via a local URL.
    # If you are running this on Hugging Face Spaces, it will be publicly available.
    demo.launch()
