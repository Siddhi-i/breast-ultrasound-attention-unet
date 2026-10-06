
import os
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr


# -----------------------------
# Load trained Attention U-Net
# -----------------------------

MODEL_PATH = "best_attention_unet.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# -----------------------------
# Preprocessing + Prediction
# -----------------------------

def predict_lesion(image):

    image = np.array(image)

    # Convert to grayscale
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Median filtering
    median = cv2.medianBlur(image, 5)

    # Sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        median,
        -1,
        kernel
    )

    # Gamma correction
    gamma = 1.2

    gamma_corrected = (
        np.power(sharpened / 255.0, gamma) * 255
    ).astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gamma_corrected)

    # Resize
    resized = cv2.resize(
        enhanced,
        (256, 256)
    )

    # Normalize
    normalized = resized.astype(
        np.float32
    ) / 255.0

    # Model input
    input_image = normalized[
        np.newaxis, ..., np.newaxis
    ]

    # Prediction
    prediction = model.predict(
        input_image,
        verbose=0
    )

    # Binary mask
    mask = (
        prediction[0, :, :, 0] > 0.5
    ).astype(np.uint8) * 255

    return resized, mask


# -----------------------------
# Gradio Interface
# -----------------------------

demo = gr.Interface(
    fn=predict_lesion,

    inputs=gr.Image(
        type="numpy",
        label="Upload Ultrasound Image"
    ),

    outputs=[
        gr.Image(
            label="Processed Ultrasound"
        ),
        gr.Image(
            label="Predicted Lesion Mask"
        )
    ],

    title="Breast Ultrasound Lesion Segmentation",

    description=(
        "Attention U-Net based breast ultrasound "
        "lesion segmentation."
    )
)


# Launch application
if __name__ == "__main__":
    demo.launch()
