# import numpy as np
# from typing import Any
# import re


# import warnings
# warnings.filterwarnings('ignore')


# class InfoRetrieval:
#     def __init__(
#         self,
#         COLLECTION_NAME: str = 'banking-products-chroma'
#     ):

#         self.COLLECTION_NAME = COLLECTION_NAME

#         self.DOCS_DIR_PATH = 'pre_data/docs/'
#         self.CHROMA_DATA_PATH: str = 'pre_data\\chroma_data\\collection_data.pkl'

#         self.is_collection_loaded = False
#         self.chroma_collection: Any = None
#         self.cross_encoder = None

#     def feedDB(self):
#         print(" - Feeding DB...")
#         token_splitted_texts = self.tokenize_split_text()
#         new_ids = [str(id_) for id_ in range(len(token_splitted_texts))]

#         self.save_chroma_collection(
#             new_ids=new_ids,
#             token_splitted_texts=token_splitted_texts
#         )
#         print(f" - Feeding DB done.\n")

#     def save_chroma_collection(
#         self,
#         new_ids: list[str],
#         token_splitted_texts: list[str]
#     ):

#         import pickle

#         print(" - Saving collection data...")
#         data = {
#             'ids': new_ids,
#             'documents': token_splitted_texts
#         }
#         with open(self.CHROMA_DATA_PATH, 'wb') as file:
#             pickle.dump(data, file)
#         print(" - Saving collection done.")

#     def tokenize_split_text(self) -> list[str]:

#         from langchain.text_splitter import (
#             RecursiveCharacterTextSplitter,
#             SentenceTransformersTokenTextSplitter
#         )

#         print(" - Tokenizing and splitting text...")
#         docs = self.read_data_from_docs()
#         all_content = '\n\n'.join([doc['content'] for doc in docs])
#         char_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1500,
#             chunk_overlap=0,
#             separators=["\n\n", "\n", ".", "!", "?", ",", ";", " ", ""],
#         )
#         token_splitter = SentenceTransformersTokenTextSplitter(
#             chunk_size=256,
#             chunk_overlap=100
#         )

#         char_splitted_text = char_splitter.split_text(text=all_content)

#         token_splitted_texts = []
#         for chunk in char_splitted_text:
#             token_splitted_texts += token_splitter.split_text(text=chunk)
#         print(f" - Tokenized and splitted text done.")
#         return token_splitted_texts

#     def read_data_from_docs(self):
#         from docx import Document
#         import os

#         print(" - Reading data from docs...")
#         docs = []
#         for filename in os.listdir(self.DOCS_DIR_PATH):
#             filepath = os.path.join(self.DOCS_DIR_PATH, filename)
#             content = ''
#             if os.path.isfile(filepath):
#                 doc = Document(filepath)
#                 full_text = []
#                 for para in doc.paragraphs:
#                     full_text.append(para.text)

#                 content = '\n'.join(full_text)

#             docs.append(
#                 {
#                     'id': filename.replace('.docx', '').title(),
#                     'content': self.clean_text(content)
#                 }
#             )
#         print(f" - Read {len(docs)} docs done.")
#         return docs

#     @staticmethod
#     def clean_text(text):
#         # Replace two or more consecutive empty lines with a single empty line
#         cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
#         return cleaned_text

#     @staticmethod
#     def word_wrap(text, width=80):

#         wrapped_text = []
#         words = text.split()
#         current_line = []

#         for word in words:
#             # If adding the new word exceeds the width, start a new line
#             if sum(len(w) + 1 for w in current_line) + len(word) > width:
#                 wrapped_text.append(' '.join(current_line))
#                 current_line = [word]
#             else:
#                 current_line.append(word)

#         # Add the last line
#         if current_line:
#             wrapped_text.append(' '.join(current_line))

#         return '\n'.join(wrapped_text)

#     def load_chroma_collection(self):

#         print(" - Loading collection...")
#         from chromadb.utils.embedding_functions import (
#             SentenceTransformerEmbeddingFunction
#         )
#         from sentence_transformers import CrossEncoder
#         import chromadb
#         import pickle

#         chroma_client = chromadb.Client()
#         embedding_function = SentenceTransformerEmbeddingFunction()

#         chroma_collection = chroma_client.get_or_create_collection(
#             name=self.COLLECTION_NAME,
#             embedding_function=embedding_function
#         )

#         with open(self.CHROMA_DATA_PATH, 'rb') as file:
#             data = pickle.load(file)

#         # Extract IDs and documents
#         new_ids = data['ids']
#         token_split_texts = data['documents']

#         chroma_collection.add(
#             ids=new_ids,
#             documents=token_split_texts
#         )

#         self.chroma_collection = chroma_collection
#         self.cross_encoder = CrossEncoder(
#             'cross-encoder/ms-marco-MiniLM-L-6-v2'
#         )

#         self.is_collection_loaded = True
#         print(" - Loading collection done.")

#     def query(self, query_text: str, n_results: int = 10):
#         if not self.is_collection_loaded:
#             self.load_chroma_collection()

#         res_docs = self.chroma_collection.query(  # type: ignore
#             query_texts=[query_text],
#             n_results=n_results
#         )['documents'][0]

#         reranked_docs = list(self.rerank_docs(res_docs, query_text))

#         return reranked_docs[0]

#     def rerank_docs(self, res_docs, query_text):

#         pairs = [[query_text, doc] for doc in res_docs]
#         scores = self.cross_encoder.predict(pairs)   # type: ignore

#         new_sorted_indexes = np.argsort(scores)[::-1]
#         new_sorted_docs = np.array(res_docs)[new_sorted_indexes]

#         if len(new_sorted_docs) > 5:
#             new_sorted_docs = new_sorted_docs[:5]
#         return new_sorted_docs


inforet = InfoRetrieval()

inforet.feedDB()

question = \
    """
     do you have account can support integration with Google Pay and Apple Pay for transactions?
    """
print('question: ', question)
print(
    inforet.query(question)
)
question = \
    """
     do you have account offer preferential exchange rates for foreign currencies?
    """

print('question: ', question)
print(
    inforet.query(question)
)
