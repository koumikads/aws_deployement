from flask import Flask, jsonify, request
import cv2
import time
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load tflite model
interpreter = tf.lite.Interpreter(model_path='best-fp16.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define endpoint to detect fish
@app.route('/detect-fish')
def detect_fish():
    # Capture image from webcam camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    # Preprocess image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img / 255.0
    img = np.expand_dims(img, 0).astype('float32')

    # Set input tensor's value
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference on the model
    interpreter.invoke()

    # Get output tensor's value
    results = interpreter.get_tensor(output_details[0]['index'])[0]

    # Filter for fish class
    fish_results = [r for r in results if r[4] == 0]

    # Return number of fish detected
    return jsonify({'num_fish': len(fish_results)})


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080,debug=True) # for local run
