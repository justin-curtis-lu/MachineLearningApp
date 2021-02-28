from transformers import pipeline
import pickle

# using pipeline API for summarization model
summarization = pipeline("summarization")

pickle.dump(summarization, open('models/text.pkl', 'wb'))