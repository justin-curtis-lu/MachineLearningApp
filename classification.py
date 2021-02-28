import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('data/iris.csv')

# Extract features and label (species)
X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])

from sklearn.preprocessing import LabelEncoder
# Transform specie names into numbers
encodedLabel = LabelEncoder()
y = encodedLabel.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
from sklearn.svm import SVC
model = SVC(kernel='linear').fit(X_train, y_train.ravel())

pickle.dump(model, open('models/iris.pkl', 'wb'))