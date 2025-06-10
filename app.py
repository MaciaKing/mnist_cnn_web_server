from flask import Flask, request, jsonify, render_template
import pdb

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template('paint.html')

    @app.route('/predict_image')
    def predict_image():
        # # if request.
        # pdb.set_trace()
        return '1'

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=True)
