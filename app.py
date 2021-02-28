import numpy as np
from flask import Flask, request, render_template
import pickle

# Create app
app = Flask(__name__)

# Retrieve our models
regressionModel = pickle.load(open('models/tips.pkl', 'rb'))
classificationModel = pickle.load(open('models/iris.pkl', 'rb'))
summarizationModel = pickle.load(open('models/text.pkl', 'rb'))

# Index Route for home
@app.route('/')
@app.route("/home")
def home():
    return render_template('index.html')

# Linear Regression
@app.route('/LR')
def LR():
    return render_template('LR.html')

# Predict
@app.route('/predictLR',methods=['POST'])
def predictLR():
    features = [int(x) for x in request.form.values()]
    featuresList = [np.array(features)]
    label = regressionModel.predict(featuresList)
    output = round(label[0], 2)
    return render_template('LR.html', prediction_text='Estimated tip Amount: $ {}'.format(output))

# Abstractive Summarization
@app.route('/AbstractiveSummarization')
def AbstractiveSummarization():
    return render_template('text.html')

# Predict
@app.route('/predictSummarization',methods=['POST'])
def predictSummarization():
    text = request.form['summary']
    output = summarizationModel(text)[0]['summary_text']
    return render_template('text.html', prediction_text='Summary: $ {}'.format(output))

# Classification
@app.route('/classification')
def classification():
    return render_template('class.html')

# Predict
@app.route('/predictClassification', methods=['POST'])
def predictClassification():
    s_length, s_width, p_length, p_width = request.form['sLength'], request.form['sWidth'],\
                                       request.form['pLength'], request.form['pWidth']
    featuresList = np.array([[s_length, s_width, p_length, p_width]])
    label = classificationModel.predict(featuresList)
    output = round(label[0], 2)
    return render_template('class.html', prediction_text='Your flower is: $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)