from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from configparser import ConfigParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--config_name", required=False, default="DEFAULT", help="DEFAULT or HISTORY")


def ask_llm(llm, question):
    response = llm(question)
    return response


def ask_rag(chain, question):
    response = chain.invoke({"input": question})
    return response


def get_rag_chain(db, system_prompt_text):
    retriever = db.as_retriever()
    system_prompt = (
        system_prompt_text + "\n\n{context}:"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    return chain


def main(llm, chain):
    while True:
        print('\n\nEnter your question:')
        question = input()
        
    
        print("\n\nLLM Response:\n")
        print(ask_llm(llm, question))
        
        print("\n\nRAG-Enhanced Response:\n")
        print(ask_rag(chain, question)['answer'])


if __name__=='__main__':
    args = vars(ap.parse_args())
    config_name = args["config_name"]
    config = ConfigParser()
    config.read('config.conf')
    config = config[config_name]

    # Load a light LLM model
    llm = OllamaLLM(model=config['llm_name'], temperature=config['llm_temperature'])
    embeddings = HuggingFaceEmbeddings(model_name=config['hugging_face_model_name'])
    db = FAISS.load_local(config['fiass_index_path'], embeddings, allow_dangerous_deserialization=True)
    chain = get_rag_chain(db, config['system_prompt_text'])

    main(llm, chain)