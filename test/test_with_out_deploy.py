import tensorflow as tf
from keras.layers import TFSMLayer
import numpy as np
from PIL import Image
import os
from tkinter.tix import IMAGE

def predict(model_to_deploy, image_path):
    try:
        # Preproces
        img = Image.open(image_path).convert("L").resize((28, 28))
        img = np.array(img).astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        pred = model_to_deploy.predict(tf.convert_to_tensor(img))
        return np.argmax(pred, axis=1)[0]

    except Exception as e:
        print(f"❌ Error {e}")
        return -1

def get_class_image(image_path):
    partes = image_path.split('_')
    ultimo = partes[-1]
    numero_str = ultimo.split('.')[0]
    return int(numero_str)

def test_image_predictions():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = base_dir + "/../modelo_completo_tf/1"
    model_to_deploy = tf.keras.Sequential([
        TFSMLayer(model_path, call_endpoint="serving_default")
    ])
    image_dir = base_dir+"/images/"

    for image_name in os.listdir(image_dir):
        y_true = get_class_image(image_name)
        predicted_class = predict(model_to_deploy, os.path.join(image_dir, image_name))
        assert y_true == predicted_class
