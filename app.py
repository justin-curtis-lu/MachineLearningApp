import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Retrieve our models
LRmodel = pickle.load(open('LR.pkl', 'rb'))
Classmodel = pickle.load(open('iris.pkl', 'rb'))
SummarizationModel = pickle.load(open('text.pkl', 'rb'))

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
    featuresList = [np.array(features)]
    label = LRmodel.predict(featuresList)
    output = round(label[0], 2)
    return render_template('LR.html', prediction_text='Estimated tip Amount: $ {}'.format(output))

@app.route('/AbstractiveSummarization')
def AbstractiveSummarization():
    return render_template('text.html')

@app.route('/predictSummarization',methods=['POST'])
def predictSummarization():
    text = request.form['summary']
    output = SummarizationModel(text)[0]['summary_text']
    return render_template('text.html', prediction_text='Summary: $ {}'.format(output))

@app.route('/classification')
def classification():
    return render_template('class.html')

@app.route('/predictClassification', methods=['POST'])
def predictClassification():
    s_length, s_width, p_length, p_width = request.form['sLength'], request.form['sWidth'],\
                                       request.form['pLength'], request.form['pWidth']
    featuresList = np.array([[s_length, s_width, p_length, p_width]])
    label = Classmodel.predict(featuresList)
    output = round(label[0], 2)
    return render_template('class.html', prediction_text='Your flower is: $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)