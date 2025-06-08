import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tkinter.tix import IMAGE

model_to_deploy = tf.keras.models.load_model("../modelo_completo_tf/1")

exclude_file = ['test_with_out_deploy.py', 'test-image.py' ]
for image_name in os.listdir():
    if image_name not in exclude_file:
        # Cargar y preprocesar imagen
        img = Image.open(image_name).convert("L").resize((28, 28))
        img = np.array(img).astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predecir
        pred = model_to_deploy.predict(img)
        predicted_class = np.argmax(pred, axis=1)[0]
        print(f'Imagen: {image_name} => Predicción: {predicted_class} => sin argmax: {pred}')
