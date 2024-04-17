from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
# CORS(app,  origins=['http://127.0.0.1:3000'])
CORS(app,  origins=['http://localhost:3000'])
# CORS(app)

model = load_model('./model_epoch_19.h5')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.post('/analyse')
def analyse_image():
    if 'image' not in request.files:
        return 'Nenhuma imagem encontrada', 400

    # image = request.files['image']
    # # Faça o que precisar com a imagem aqui

    # return 'Imagem recebida com sucesso', 200

    image_file = request.files['image']
    image = Image.open(image_file)
    image = image.resize((100, 100))
    image = np.array(image) / 255.0  # Normalizar os valores dos pixels
    image = np.expand_dims(image, axis=0)  # Adicione uma dimensão extra para criar um lote de uma única imagem

    # Predict using the model
    predictions = model.predict(image)
    print(predictions)

    # Converta as previsões em uma resposta JSON
    response = {'predictions': predictions.tolist()}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)