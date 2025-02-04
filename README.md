# llm_rag_example

This repository provides a simple example of using **LLM + RAG (Retrieval-Augmented Generation)** to answer questions based on uploaded documents. By adding PDFs to a folder, this example generates embeddings for efficient document retrieval and answers questions related to the content.

This project uses FAISS for fast similarity search and a Hugging Face model to generate embeddings for document retrieval.

## Setup

Make sure you have **Python 3.9+** installed. Then, install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Configuration

You can configure the path to the PDF files in the config.conf file.


## Step 1: Build Embeddings
First, generate and save embeddings for the PDFs in specified folder in config by running:

```bash
python build_embeddings.py --config_name CONFIG_NAME
```

## Step 2: Ask Questions
Once embeddings are created, you can query the system. Use the following command to ask questions:

```bash
python run.py --config_name CONFIG_NAME
```