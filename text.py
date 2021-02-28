from transformers import pipeline
import pickle
# using pipeline API for summarization task
summarization = pipeline("summarization")
pickle.dump(summarization, open('text.pkl', 'wb'))