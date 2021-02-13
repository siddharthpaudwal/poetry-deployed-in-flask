from flask import Flask,render_template,request
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)
token = pickle.load(open('tokenizer', 'rb'))
model = keras.models.load_model('my_model')
poetry_length = 10
def generate_poetry(seed_text,n_lines):
    fin_list = []
    fin_list.append(seed_text)
    for i in range(n_lines):
        text = []
        for _ in range(poetry_length):
          encoded = token.texts_to_sequences([seed_text])
          encoded = pad_sequences(encoded, maxlen=19, padding='pre')

          y_pred = np.argmax(model.predict(encoded), axis=-1)

          predicted_word = ""
          for word, index in token.word_index.items():
            if index == y_pred:
              predicted_word = word
              break

          seed_text = seed_text + ' ' + predicted_word
          text.append(predicted_word)
        seed_text = text[-1]
        text = ' '.join(text)
        fin_list.append(text)
    return fin_list
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text = request.form['text_field']
    res = []
    res = generate_poetry(text,5)
    return render_template('index.html', prediction_text=res)
if __name__=="__main__":
    app.run(debug=True)