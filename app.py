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
@app.route('/LinearRegression')
def LinearRegression():
    return render_template('LR.html')

# Predict
@app.route('/PredictRegression',methods=['POST'])
def PredictRegression():
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
@app.route('/PredictSummarization',methods=['POST'])
def PredictSummarization():
    text = request.form['summary']
    output = summarizationModel(text)[0]['summary_text']
    return render_template('text.html', prediction_text=output)

# Classification
@app.route('/Classification')
def Classification():
    return render_template('class.html')

# Predict
@app.route('/PredictClassification', methods=['POST'])
def PredictClassification():
    s_length, s_width, p_length, p_width = request.form['sLength'], request.form['sWidth'],\
                                       request.form['pLength'], request.form['pWidth']
    featuresList = np.array([[s_length, s_width, p_length, p_width]])
    label = classificationModel.predict(featuresList)
    output = round(label[0], 2)
    return render_template('class.html', species=output)

if __name__ == "__main__":
    app.run(debug=True)