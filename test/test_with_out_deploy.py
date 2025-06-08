import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tkinter.tix import IMAGE

base_dir = os.path.dirname(os.path.abspath(__file__))

model_to_deploy = tf.keras.models.load_model(base_dir+"/../modelo_completo_tf/1")


# Ruta de las imágenes
image_dir = base_dir+"/images/"

# Iterar sobre las imágenes
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)

    try:
        # Cargar y preprocesar imagen
        img = Image.open(image_path).convert("L").resize((28, 28))
        img = np.array(img).astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predecir
        pred = model_to_deploy.predict(tf.convert_to_tensor(img))
        predicted_class = np.argmax(pred, axis=1)[0]
        print(f'Imagen: {image_name} => Predicción: {predicted_class} => sin argmax: {pred}')
    except Exception as e:
        print(f"❌ Error con {image_name}: {e}")
