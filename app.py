from flask import Flask, request, jsonify, render_template
from io import BytesIO
import json
from PIL import Image
import io
import numpy as np
import base64
import requests
from tensorflow.keras.preprocessing import image
import pdb

def send_image_to_tf_serving():
    None

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template('paint.html')

    @app.route('/predict_image', methods=['POST'])
    def predict_image():
        data = json.loads(request.data)
        base64_str = data['image'].split(",")[1]
        image_bytes = base64.b64decode(base64_str)

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((28, 28))  # Cambia tamaño según tu modelo
        img_array = np.array(image) / 255.0

        # 5. Añadir dimensión batch
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # 6. Crear payload JSON para TF Serving
        payload = {
            "instances": img_array.tolist()
        }

        r = requests.post('http://tensorflow-serving:8501/v1/models/modelo_completo_tf:predict', json=payload)

        pdb.set_trace()

        # Decoding results from TensorFlow Serving server
        pred = json.loads(r.content.decode('utf-8'))

        return(np.array(pred['predictions'])[0] > 0.4).astype(np.int)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=True)
