from transformers import pipeline

# Function to download the model weigth locally
def downloadTranslator(name, translator):
    translator.save_pretrained(name)

# Funtion to load the model from local weights
def getTranslator(path):
    return pipeline(task = 'translation', model=path)

# Function to get the translation
def getTranslation(body, translator):
    result = translator(body)[0]['translation_text']
    return result
