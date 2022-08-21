# Cloud_Computing_ENSAE

## NLP application

The purpose of this project is to create a HTML application with 3 differents Python functions (tabs):
- **Wikipedia scrapping**
- **Text translation** 
- **Text summarization**
  

## Installation

To install the package in local environment:

```bash
pip install -r requirements.txt
flask run
```

## Description

### Wikipedia scrapping

It allows the possibility to scrap a Web page from an URL or to get the Wikipedia content from a word.
Can be used with
  - def main_scrape(what,lang,max_len)          --->       to scrape a web page from a url or a word

### Text translation

It allows the possibility to translate a text from French to English or English to French.
3 functions can be used:
  - def downloadTranslator(name, translator)     --->       to download paramenters of the model locally once
  - def getTranslator(path)                      --->       to load the model from local weights
  - def getTranslation(body, translator)         --->       to get the translation


## Unit tests

To run unit tests:

```bash
python -m unittest test_unit.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Contributors

- Victoire DAGNEAU DE RICHECOUR
- Yann DE COSTER
- Thomas DOUCET
- Paul PETIT
- Ivanna SAVONIK

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)
