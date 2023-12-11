from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
import io
import os

app = Flask(__name__)
CORS(app, resources={"*": {"origins": "*"}})

interpreter, input_details, output_details, labels = None, None, None, None


# Load the TensorFlow Lite model
def init(plant_name):
    global interpreter, input_details, output_details, labels
    path = f"{os.getcwd()}/models/{plant_name}.tflite"
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    label_path = f"{os.getcwd()}/models/{plant_name}.txt"
    with open(label_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]


# Function to preprocess the input image
def preprocess_image(base64_image):
    try:
        # Split the base64 string to get the actual image data
        _, image_data = base64_image.split(";base64,")

        # Decode base64 image and convert to NumPy array
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data))
        if base64_image.startswith("data:image/png;base64,"):
            image = image.convert("RGB")
        image = image.resize(
            (input_details[0]["shape"][2], input_details[0]["shape"][1])
        )
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image.astype(np.float32)
    except Exception as e:
        raise ValueError("Error preprocessing image: {}".format(str(e)))


# Function to perform inference
def run_inference(image):
    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


@app.route("/", methods=["GET"])
def hello():
    return "Hello World!"


# Flask route for image classification
@app.route("/classify", methods=["POST"])
def classify_image():
    data = request.get_json()

    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    if "plant" not in data:
        return jsonify({"error": "No plant name provided"}), 400

    init(data["plant"])

    try:
        # Preprocess the base64-encoded image
        input_image = preprocess_image(data["image"])

        # Run inference
        predictions = run_inference(input_image)

        # Get the top prediction
        top_prediction = np.argmax(predictions)
        confidence = predictions[0, top_prediction]

        return jsonify(
            {
                "class_name": labels[float(confidence) <= 0.2]
                if data["plant"] == "pepper_bell"
                else labels[int(top_prediction)],
                "confidence": float(confidence),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
