from flask import Flask, request, jsonify
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from io import BytesIO

app = Flask(__name__)
model = tf.keras.models.load_model('my_fruit_classification_model.h5')

def preprocess_image(image):
    image = Image.open(BytesIO(image)).resize((100, 100))
    img_array = img_to_array(image)
    img_array = np.array([img_array]) / 255.0
    return img_array

@app.route('/', methods=['POST'])
def predict():
    image_data = request.get_json()['image']
    image = base64.b64decode(image_data)
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=-1)[0]
    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)