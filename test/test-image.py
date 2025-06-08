import requests
import numpy as np
from PIL import Image
import os
from tkinter.tix import IMAGE

exclude_file = ['test-image.py', 'test_with_out_deploy']
for image_name in os.listdir():
    if image_name not in exclude_file:
        # Cargar y preprocesar imagen
        img = Image.open(image_name).convert("L").resize((28, 28))
        img = np.array(img).astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Enviar a servidor TensorFlow Serving
        data = {"instances": img.tolist()}
        url = "http://localhost:8501/v1/models/modelo_completo_tf:predict"

        response = requests.post(url, json=data)
        predictions = response.json()

        print(f"[{image_name}] - class: {np.argmax(predictions['predictions'])} - prediction: {predictions}")
        # printf(f"[] Predicción: {np.argmax(predictions["predictions"][0])}" )
