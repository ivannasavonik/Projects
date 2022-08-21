from flask import Flask, render_template, request
from transformers import pipeline
from scrape import main_scrape
from translate import *
from summary import *
import os.path
from os import path

if __name__ == "__main__":
    app.run(debug=True, port=8000)

# To download the model weigth locally the first time if not already downloaded
if not path.exists("translator_EF"):
    downloadTranslator(name='translator_EF', translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr"))
if not path.exists("translator_FE"):
    downloadTranslator(name='translator_FE', translator=pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en"))
if not path.exists("summarisation"):
    downloadSummarization(name='summarisation',  model= pipeline("summarization"))


# Load the models
translator_EF = getTranslator("translator_EF")
translator_FE = getTranslator("translator_FE")
model = getSummarization("summarisation")

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def affichage():
    buttonId = "Accueil_button"

    if request.method == 'POST':
        scrap_input = request.form.get('scrap_in')
        trad_input = request.form.get('trad_in')
        trad_checkbox = request.form.get('translate_fr_to_en')
        res_input = request.form.get('res_in')

        scrap_output, trad_output, res_output = "", "", ""
        if scrap_input is not None:
            if request.form.get('scrapping_language') == 'french':
                scrap_output = main_scrape(scrap_input, 'fr', 1500)
            else:
                scrap_output = main_scrape(scrap_input, 'en', 1500)
            buttonId = "Scrapping_button"
        if trad_input is not None:
            if trad_checkbox == 'fr_to_en':
                print(trad_checkbox)
                trad_output = getTranslation(trad_input, translator=translator_FE)
            else:
                trad_output = getTranslation(trad_input, translator=translator_EF)
            buttonId = "Traduction_button"
        if res_input is not None:
            res_output = summarization(res_input, model=model)
            buttonId = "Résumé_button"

        outputs = [scrap_output, trad_output, res_output]
        return render_template('index.html', output=outputs, buttonId=buttonId)
    return render_template('index.html', output=[], buttonId=buttonId)