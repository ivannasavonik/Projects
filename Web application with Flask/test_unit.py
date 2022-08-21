import unittest
from nltk.tokenize import TweetTokenizer
from scrape import *
from translate import *
from summary import *
import os.path
from os import path


class MyTestCase(unittest.TestCase):
    # Test remove reférence
    def test_remove_ref(self):
        test = "With over 8 million players, Fantasy Premier League [1] is the biggest Fantasy Football game in the world [ 2]."
        test = TweetTokenizer().tokenize(test)
        expected_result = ['With', 'over', '8', 'million', 'players', ',', 'Fantasy', 'Premier', 'League', 'is', 'the',
                           'biggest', 'Fantasy', 'Football', 'game', 'in', 'the', 'world', '.']
        self.assertEqual(remove_reference(test), expected_result)

    # Test transformation des tokens à une phrase
    def test_token_phrase(self):
        une_phrase = 'je suis une phrase mais bientot des tokens'
        res = token_to_txt(TweetTokenizer().tokenize(une_phrase))
        self.assertEqual(res, une_phrase)

    # Test remove url
    def test_remove_url(self):
        une_phrase = ' le site web est https://fr.wikipedia.org/wiki/Python_(serpent). Oops il est plus là'
        expected_result = 'le site web est. Oops il est plus là'
        res = token_to_txt(remove_url(TweetTokenizer().tokenize(une_phrase)))
        self.assertEqual(res, expected_result)

    # Test remove hashtags
    def test_remove_hashtags(self):
        une_phrase = "Cloud Computing c'est #WAOUH"
        expected_result = "Cloud Computing c'est WAOUH"
        res = token_to_txt(remove_hashtags(TweetTokenizer().tokenize(une_phrase)))
        self.assertEqual(res, expected_result)

    # Rien a supprimer
    def test_clean_wiki_fr_1(self):
        test_clean = "Elle comptait à la fin de 2019 près de 266 millions  de clients dans le monde, des chiffres en hausse par rapport à ceux affichés en 2018. En 2019, l'entreprise est leader ou second opérateur dans 75% des pays européens où elle est implantée et dans 83% des pays en Afrique et au Moyen-Orient."
        expected_result = test_clean
        res = clean_wiki_french(test_clean)
        self.assertEqual(res, expected_result)

    # Nettoyage nécéssaire
    def test_clean_wiki_fr_2(self):
        test_clean = 'Pages pour les contributeurs déconnectés en savoir plus Pour les articles homonymes, voir Orange. Orange est une société française de télécommunications.'
        expected_result = 'Orange est une société française de télécommunications.'
        res = clean_wiki_french(test_clean)
        self.assertEqual(res, expected_result)

    # input : Mot Francais - correspondance directe
    def test_scrap_1(self):
        test = main_scrape('Docker', 'fr', 300)
        expected_result = 'Un docker(/ dɔ. kɛʁ/) ou débardeur est un ouvrier portuaire, travaillant dans les docks, employé au chargement et déchargement des navires arrivant ou quittant le port. Il existe différents noms communs pour désigner la profession qui exerce cette activité selon les régions du monde.'
        self.assertEqual(test, expected_result)

    # URL autre que wiki
    def test_scrap_2(self):
        test = main_scrape('https://python.doctor/', 'fr', 200)
        expected_result = 'Python est un langage de programmation.'
        self.assertEqual(test, expected_result)

    # URL wiki
    def test_scrap_3(self):
        test = main_scrape('https://fr.wikipedia.org/wiki/Rhinoceros', 'fr', 200)
        expected_result = 'Cet article est une ébauche concernant un mammifère. Vous pouvez partager vos connaissances en l ’ améliorant( comment?) selon les recommandations du projet zoologie.'
        self.assertEqual(test, expected_result)

    # Absence d'article wiki
    def test_scrap_4(self):
        test = main_scrape('fuyez pauvres fous', 'fr', 2000)
        expected_result = "Il n'existe pas de page wikipédia"
        self.assertEqual(test[:33], expected_result)

    # Mot : correspondance indirect wiki carr ambiguité
    def test_scrap_5(self):
        res = main_scrape('Docker', 'en', 300)
        expected_result = 'Docker is a set of platform as a service( PaaS) products that use OS-level virtualization to deliver software in packages called containers.'
        self.assertEqual(res, expected_result)

    def test_get_translation_EF(self):
        # To download the model weigth locally the first time if not already downloaded
        if not path.exists("translator_EF"):
            downloadTranslator(name='translator_EF', translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr"))
        # Load the model
        translator_EF = getTranslator("translator_EF")
        res = getTranslation("Hi, chocolate is for people with arms.", translator=translator_EF)
        expected_result = "Salut, le chocolat, c'est pour les gens avec des bras."
        self.assertEqual(res, expected_result)

    def test_get_translation_FE(self):
        # To download the model weigth locally the first time if not already downloaded
        if not path.exists("translator_FE"):
            downloadTranslator(name='translator_FE', translator=pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en"))
        # Load the model
        translator_FE = getTranslator("translator_FE")
        res = getTranslation("Salut, le chocolat, c'est pour les gens avec des bras.", translator = translator_FE)
        expected_result = "Hey, chocolate is for people with arms."
        self.assertEqual(res, expected_result)
    
    def test_summarize(self):
        # To download the model weigth locally the first time if not already downloaded
        if not path.exists("summarisation"):
            downloadSummarization(name='summarisation', model= pipeline("summarization"))
        # Load the model
        model = getSummarization("summarisation")
        text="Equitable access to safe and effective vaccines is critical to ending the COVID-19 pandemic, so it is hugely encouraging to see so many vaccines proving and going into development. WHO is working tirelessly with partners to develop, manufacture and deploy safe and effective vaccines.Safe and effective vaccines are a game-changing tool: but for the foreseeable future we must continue wearing masks, cleaning our hands, ensuring good ventilation indoors, physically distancing and avoiding crowds. Being vaccinated does not mean that we can throw caution to the wind and put ourselves and others at risk, particularly because research is still ongoing into how much vaccines protect not only against disease but also against infection and transmission.See WHO’s landscape of COVID-19 vaccine candidates for the latest information on vaccines in clinical and pre-clinical development, generally updated twice a week. WHO’s COVID-19 dashboard, updated daily, also features the number of vaccine doses administered globally.But it’s not vaccines that will stop the pandemic, it’s vaccination. We must ensure fair and equitable access to vaccines, and ensure every country receives them and can roll them out to protect their people, starting with the most vulnerable."
        res= summarization(text, model=model)
        expected_result=' WHO is working tirelessly with partners to develop, manufacture and deploy safe and effective vaccines . Equitable access to vaccines is critical to ending the COVID-19 pandemic . But for the foreseeable future we must continue wearing masks, cleaning our hands, ensuring good ventilation indoors, physically distancing and avoiding crowds .'
        self.assertEqual(res, expected_result)



if __name__ == '__main__':
    #print('non')
    unittest.test_remove_ref()
    #unittest.test_scrap_1()
    #unittest.test_scrap_2()
    #unittest.test_scrap_3()