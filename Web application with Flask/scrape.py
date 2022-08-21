import string
from nltk.tokenize import TweetTokenizer
import requests
from bs4 import BeautifulSoup
import re


def get_text_from_url(url: str, max_len: int = -1):
    '''
    Returns the text of the web page informed

    :param url: str
           max_len: int

    :return: list with all the text of the page
    '''

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    str_output = ''
    actual_len = 0
    all_text = soup.find_all('p')
    for i in range(len(all_text)):
        actual_len += len(all_text[i].get_text())
        if max_len > 0 and max_len < actual_len:
            where = all_text[i].get_text().find('.')
            if where == -1:  # Si il n'y pas de point alors on renvoie la dernière zone de texte entière
                str_output += all_text[i].get_text() + ' '
            else:  # On s'arrête au prochain point
                str_output += all_text[i].get_text()[:where + 1]
            break
        else:
            str_output += all_text[i].get_text() + ' '
    # Absence de texte dans l'URL : exemple google_image ou site web protégé
    if actual_len==0:
        return "L'URL renseigné ne permet pas de récupérer du texte (absence de texte ou site protégé)"
    else:
        return str_output


def remove_hashtags(tokens):
    """

    :param tokens:
    :return: tokens without hashtags
    """
    tokens = map(lambda x: x.replace('#', ''), tokens)
    return list(tokens)


def remove_url(tokens):
    """

    :param tokens:
    :return: tokens without url
    """
    tokens = filter(lambda x: "http" not in x, tokens)
    return list(tokens)


def remove_html(tokens):
    """

    :param tokens:
    :return: tokens without html
    """
    tokens = filter(lambda x: x[0] + x[-1] != '<>', tokens)
    return list(tokens)


def remove_reference(tokens):
    """

    :param tokens:
    :return: tokens without reference ([3])
    """
    while '[' in tokens:
        start = tokens.index('[')
        # Si il n'y a plus de fin de référence alors break
        if not ']' in tokens:
            break
        else:
            end = tokens.index(']')
        tokens = tokens[:start] + tokens[end + 1:]
    return tokens


def token_to_txt(tokens):
    """
    Transform token to text
    :param tokens:
    :return:
    """
    res = tokens[0]
    for tok in tokens[1:]:
        if tok in string.punctuation + "'":
            res += tok
        else:
            res = res + ' ' + tok
    return res

def clean_wiki_french(text):
    """
    Remove the first sentence from the French wikipedia
    if there is a reference to unconnected contributors or "homonymes"

    :param text:
    :return:
    """

    if text is not None:
        if '.' in text: #Pas de phrase
            phrase1=text[:text.index('.')]
            if 'contributeurs' in phrase1 or 'articles homonymes' in phrase1 :
                text=text[text.index('.')+2:]
            return text
        else:
            return text
    else:
        return 0

def get_wiki_from_word_research_page(word:str,lang='fr'):
    """
    Search a wikipedia page from a word or group of words using the search page

    :param word:
    :param lang: 'en' or 'fr
    :return: dic, first_url --> dictionary with every wikepedia pages with
    a reference to the argument word and the first url
    """
    if lang=='fr':
        start_url='https://fr.wikipedia.org'
        search_start='https://fr.wikipedia.org/w/index.php?title=Sp%C3%A9cial:Recherche&search='
        search_end='&profile=advanced&fulltext=1&ns0=1'
    elif lang=='en':
        start_url='https://en.wikipedia.org'
        search_start='https://en.wikipedia.org/w/index.php?search='
        search_end='&title=Special:Search&profile=advanced&fulltext=1&ns0=1'
    url_search=search_start+word+search_end
    page = requests.get(url_search)
    soup = BeautifulSoup(page.content, 'html.parser')
    res=soup.find("ul", attrs={"class": "mw-search-results"})
    dic={}
    for rep in res.find_all('a'):
        dic[rep.get('title')]=rep.get('href')
        # Récupérer un URL
    for key in dic.keys():
        if key.lower() != word.lower():
            fin_url = dic[key]
            break
    first_url = start_url + fin_url
    return dic,first_url


def get_text_wiki_from_word(word_ref: str, lang: str, max_len: int):
    """
    Adding the search word in the url directly and
    identifying if the page exists or if it is an ambiguous page

    :param word_ref:
    :param lang:
    :param max_len:
    :return:
    """
    if lang == 'en':
        url = 'https://en.wikipedia.org/wiki/' + word_ref
        ambigu = 'Disambiguation pages'
        len_amnbigu = len(ambigu)
    elif lang == 'fr':
        url = 'https://fr.wikipedia.org/wiki/' + word_ref
        ambigu = 'Homonymie'
        len_amnbigu = len(ambigu)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    if lang == 'en':
        check = soup.find("a", attrs={"title": re.compile("^Category")})

    elif lang == 'fr':
        check = soup.find("a", attrs={"title": re.compile("^Catégorie")})  # Vérifie si la page existe
        check_homo = soup.find("a", attrs={"title": re.compile("^Catégorie:Hom")})  # Si c'est un cas d'homonyme

    if check is None:  # Page non trouvé
        if lang == 'en':
            la_langue='anglais'
        else:
            la_langue='français'

        return "Il n'existe pas de page wikipédia en " + la_langue + ' faisant référence à '+word_ref

    else:  # Mot ambigu
        is_ambigu = False
        if lang == 'en':
            res = check.get('title')
            if res[-len_amnbigu:] == ambigu:  # Page ambigue
                is_ambigu = True
                # Extraire première page
        elif lang == 'fr':
            if check_homo is not None:
                is_ambigu = True
        if is_ambigu:
            dic_all, first_link = get_wiki_from_word_research_page(word_ref, lang)
            res = get_text_from_url(first_link, max_len)

            pipe_clean = [TweetTokenizer().tokenize, remove_html, remove_url, remove_hashtags,token_to_txt,
                          remove_reference,clean_wiki_french]
            for func in pipe_clean:
                res = func(res)
            return res
        else:  # La page est bonne, on peut scraper
            res = get_text_from_url(url, max_len)
            pipe_clean = [TweetTokenizer().tokenize, remove_html, remove_url, remove_hashtags,token_to_txt,
                          remove_reference,clean_wiki_french]

            for func in pipe_clean:
                res = func(res)
            return res

def main_scrape (what,lang,max_len):
    """
    main function allowing to scrape a web page from a url or a word

    :param what: url or word
    :param lang: language of research if what is a word
    :param max_len: maximum length of the output
    :return: text from a wiki page
    """
    if 'http' in what: # URL Case
        res=get_text_from_url(what,max_len)
        pipe_clean=[TweetTokenizer().tokenize,remove_html,remove_url,remove_hashtags,token_to_txt,
                    remove_reference,clean_wiki_french]
        for func in pipe_clean:
            res=func(res)
        return res
    else : # Word Case
        res =get_text_wiki_from_word(what,lang,max_len)
        return res

