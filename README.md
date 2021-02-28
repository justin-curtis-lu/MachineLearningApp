# MachineLearningApp

This application allows the user to provide input for a linear regression, classification, and text summarization model. The framework was done using Flask and the models created using sklearn and transformers module. Pickle was used to save the models and retrieved when the application is served. The main reason I wanted to work on this project was to get experience deploying ML models on a web framework.

## Getting Started

Clone the repository

### Installing

If using a PyCharm virtual enviornment the modules may not install smoothly and you may want to use Anaconda for 
some of the ML modules. 
All of the modules found in the requirements.txt file are needed for this project. In your virtual enviornment type

```
pip install -r requirements.txt
```
### Models

Make sure to run summarization.py in order to create the pkl file for the model. Since the model is large and isnt
on git you have to create it manually. The other 2 models are already created and in the models directory.

```
py summarization.py
```

## Deployment

This application can be deployed using Flask and served on the local host

```
py app.py
```

## Built With

* [Flask](https://flask.palletsprojects.com/en/1.1.x/) - The web framework used
* [Scikit-learn](https://scikit-learn.org/stable/) - For Linear Regression and Classification models
* [Transformers](https://pypi.org/project/transformers/) - For Abstractive Summarization model
