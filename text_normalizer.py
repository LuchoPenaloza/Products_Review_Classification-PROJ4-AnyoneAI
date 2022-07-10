import re
import nltk
import spacy
import unicodedata

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer

nltk.download('stopwords')

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')


def remove_html_tags(text):
    # Put your code
    patt_remove_html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(patt_remove_html, '', text)
    return text


def stem_text(text):
    # Put your code
    text_tokenize = tokenizer.tokenize(text)
    text_new = [nltk.porter.PorterStemmer().stem(w) for w in text_tokenize]    
    text = " ".join(text_new)
    return text


def lemmatize_text(text):
    # Put your code
    text_doc = nlp(text)
    text_new = [token.lemma_ for token in text_doc]
    text = " ".join(text_new)
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    # Put your code
    for word, replacement in CONTRACTION_MAP.items():
        text = text.replace(word, replacement)
    return text


def remove_accented_chars(text):
    # Put your code
    nfkd_form = unicodedata.normalize('NFKD', text)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    text = only_ascii.decode('UTF-8')
    return text


def remove_special_chars(text, remove_digits=False):
    # Put your code
    pattern_1 = r"[^a-zA-Z0-9 ]+"
    text = re.sub(pattern_1, '', text)
    if remove_digits:
        pattern_2 = r"[\d]"
        text = re.sub(pattern_2, '', text)
    return text


def remove_stopwords(text, is_lower_case=True, stopwords=stopword_list):
    # Put your code
    if is_lower_case:
        text = text.lower()
    text_tokenize = tokenizer.tokenize(text)
    text_new = [w for w in text_tokenize if not w in stopword_list]    
    text = " ".join(text_new)
    return text


def remove_extra_new_lines(text):
    # Put your code
    text = re.sub('\n', ' ', text.strip())
    return text


def remove_extra_whitespace(text):
    # Put your code
    text = re.sub(' +', ' ', text.strip())
    return text
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
