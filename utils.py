import numpy as np
from PIL import Image
import tensorflow as tf

# Load the TensorFlow Lite model
def load_model(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess image
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Predict class from image
def predict(image: Image.Image, class_names: list, interpreter) -> tuple:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    processed_image = preprocess_image(image)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = int(np.argmax(output_data))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(output_data))

    return predicted_class, confidence
