import pandas as pd
import pickle

# Using pandas read in our data set
dataset = pd.read_csv('tips.csv')

# For our linear regression, we will designate all columns excluding (tip)
# To be our X variables
x_variables = [0, 2, 3, 4, 5, 6]
X = dataset.iloc[:, x_variables]

# Create conversion functions for all categorical columns
def convert_sex(word):
    words = { 'Female': 0, 'Male': 1}
    return words[word]
def convert_smoker(word):
    words = { 'No': 0, 'Yes': 1}
    return words[word]
def convert_day(word):
    words = { 'Sun':0 , 'Sat': 1, 'Thur': 2, 'Fri': 3}
    return words[word]
def convert_time(word):
    words = { 'Dinner': 0, 'Lunch': 1}
    return words[word]

# Apply conversions
X['sex'] = X['sex'].apply(lambda x: convert_sex(x))
X['smoker'] = X['smoker'].apply(lambda x: convert_smoker(x))
X['day'] = X['day'].apply(lambda x: convert_day(x))
X['time'] = X['time'].apply(lambda x: convert_time(x))

# Set Tips to be our Y variable
y = dataset.iloc[:, 1]

# Import the linear regression model from sklearn module
from sklearn.linear_model import LinearRegression

# Create the model object
lrModel = LinearRegression()

# Train the model with our X and Y variables
lrModel.fit(X, y)

# Use dump attribute on model to store it for our app
pickle.dump(lrModel, open('model.pkl', 'wb'))
