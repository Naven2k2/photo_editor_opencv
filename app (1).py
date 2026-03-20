import cv2
import numpy as np
import gradio as gr
from PIL import Image

def process_image(image, width, height, brightness, contrast, filter_option):
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (width, height))

    # Brightness & Contrast
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    output = img.copy()

    # Filters
    if filter_option == "None":
        output = img.copy()

    elif filter_option == "Grayscale":
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    elif filter_option == "Blur":
        output = cv2.GaussianBlur(output, (15, 15), 0)

    elif filter_option == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        output = cv2.filter2D(output, -1, kernel)

    elif filter_option == "Warm":
        output = np.clip(output + [10, 0, -10], 0, 255).astype(np.uint8)

    elif filter_option == "Portrait Blur":
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        mask = cv2.GaussianBlur(gray, (25, 25), 0)
        mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)

        blurred = cv2.GaussianBlur(output, (35, 35), 0)
        output = (output * mask[..., None] + blurred * (1 - mask[..., None])).astype(np.uint8)

    elif filter_option == "Edge Detection":
        output = cv2.Canny(output, 100, 200)

    elif filter_option == "Cartoon":
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9, 9
        )
        color = cv2.bilateralFilter(output, 9, 300, 300)
        output = cv2.bitwise_and(color, color, mask=edges)

    # Convert BGR → RGB (IMPORTANT)
    if len(output.shape) == 3:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    return output


interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(100, 1000, value=500, label="Width"),
        gr.Slider(100, 1000, value=500, label="Height"),
        gr.Slider(-100, 100, value=0, label="Brightness"),
        gr.Slider(1.0, 3.0, value=1.0, label="Contrast"),
        gr.Dropdown(
            ["None", "Grayscale", "Blur", "Sharpen", "Warm",
             "Portrait Blur", "Edge Detection", "Cartoon"],
            label="Select Filter"
        )
    ],
    outputs=gr.Image(type="numpy", label="Edited Image"),
    title="📸 Photo Editor using OpenCV",
    description="Upload an image, adjust settings, apply filters, and download your edited image."
)

interface.launch()