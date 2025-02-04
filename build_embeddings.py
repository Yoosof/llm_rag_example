import os
from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from configparser import ConfigParser
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--config_name", required=True, help="name of the config")


def read_docs(books_path):
    docs = []
    for filename in os.listdir(books_path):
        print('******************* filename: ', filename)
        loader = PDFMinerLoader(os.path.join(books_path, filename))  # Add actual books in this file
        docs += loader.load()
    return docs

def split_docs(docs, chunk_size=500, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks


if __name__ == '__main__':
    args = vars(ap.parse_args())
    config_name = args["config_name"]
    config = ConfigParser()
    config.read('config.conf')
    config = config[config_name]

    embeddings = HuggingFaceEmbeddings(model_name=config['hugging_face_model_name'])
    docs = read_docs(config['books_path'])
    chunks = split_docs(docs)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(config['fiass_index_path'])


