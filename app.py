import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Open our pickled model
LRmodel = pickle.load(open('model.pkl', 'rb'))

# Index Route
@app.route('/')
@app.route("/home")
def home():
    return render_template('index.html')

@app.route('/LR')
def LR():
    return render_template('LR.html')

# Route for tip prediction
@app.route('/predictLR',methods=['POST'])
def predictLR():
    features = [int(x) for x in request.form.values()]
    label = [np.array(features)]
    prediction = LRmodel.predict(label)
    output = round(prediction[0], 2)
    return render_template('LR.html', prediction_text='Estimated tip Amount: $ {}'.format(output))

@app.route('/classification')
def classification():
    return render_template('class.html')



if __name__ == "__main__":
    app.run(debug=True)