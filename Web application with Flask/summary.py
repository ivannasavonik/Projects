from transformers import pipeline 

# Function to download the model weigth locally
def downloadSummarization(name, model):
    model.save_pretrained(name)

# Funtion to load the model from local weights
def getSummarization(path):
    return pipeline(task = 'summarization', model=path)

# Function to get the summarization
def summarization(body , model):
    result = model(body, min_length=25, max_length=150)[0]['summary_text']
    return result