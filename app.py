import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Open our pickled model
model = pickle.load(open('model.pkl', 'rb'))

# Index Route
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict',methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    label = [np.array(features)]
    prediction = model.predict(label)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Estimated tip Amount: $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)