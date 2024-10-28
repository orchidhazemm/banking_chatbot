from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import lru_cache
from deep_translator import GoogleTranslator

import re

def get_current_time():
    current_time = datetime.now().strftime('%I:%M %p')
    return current_time


def advanced_string_search(statement: str, big_string: str) -> bool:
    # Function to clean text by removing non-alphabetical characters
    if not statement or not big_string:
        return True

    # Normalize and clean the strings
    normalized_statement = statement.strip().lower()
    normalized_big_string = big_string.strip().lower()

    # Split the normalized and cleaned statement into individual words
    words = normalized_statement.split()

    # Check if all words are present in the big string (in any order)
    res = [re.search(re.escape(word), normalized_big_string) for word in words]

    print('-'*50)
    print(f"words: {words}")
    print(f"res: {res}")

    ratio = res.count(None) / len(words)
    print(f"ratio: {ratio}")
    print('-'*50)
    return res.count(None) == 0


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


@lru_cache
def stem(word):
    return stemmer.stem(word.lower())


cleaned_texts = dict()


def clean_text(text: str) -> str:

    global cleaned_texts
    if text in cleaned_texts:
        return cleaned_texts[text]

    # Clean the text by removing non-alphabetical characters
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text into words
    words = word_tokenize(cleaned_text)

    # Filter out stopwords and lemmatize the words
    filtered_words = [
        stem(word)
        for word in words if word.lower() not in stop_words
    ]

    # Join the words back into a single string
    result = ' '.join(filtered_words)

    cleaned_texts[text] = result

    return result


def translate(text, dest):

    if dest not in ['en', 'ar']:
        raise ValueError(
            "Target language must be 'en' for English or 'ar' for Arabic.")

    translated = GoogleTranslator(
        source='auto',
        target=dest).translate(text)
    return translated


def get_arabic_markdown(markdown_text):
    new_markdown_text = \
        f"""
        <div dir="rtl" style="text-align: right;">
        {markdown_text}
        """
    return new_markdown_text
