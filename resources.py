import logging
import pathlib
from typing import Iterator
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.llms.ollama import Ollama
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.document_loaders import WebBaseLoader, TextLoader

import requests
from langchain_community.embeddings import OllamaEmbeddings

from langchain import hub
from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)


def create_plan(weekly_kilometres: int, weekly_runs: int):
    rag_chain(
        f"Please write me a running plan. I want to run exactly {weekly_kilometres}km per week, spread out over {weekly_runs} runs. Format should strictly be 'Day 1- run type, distance (km). Day 2- run type, distance (km), etc. You must not go over weekly mileage specified."
    )


# Load content of blog
TXT_DOC_PATH = pathlib.Path("blog_text.txt")
loader = TextLoader(file_path=(TXT_DOC_PATH))

# below code can be used to scrape internet to populate txt file
# will want to automate full process eventually

# html_doc = requests.get(
#     "https://run.outsideonline.com/training/rules-when-setting-up-weekly-running-schedule/"
# )

# soup = BeautifulSoup(
#     html_doc.text,
#     "html.parser",
#     parse_only=SoupStrainer(
#         "article",
#         attrs={"class": "l-container l-article l-article--post u-spacing--triple"},
#     ),
# )

# # Get rid of tags
# soup = soup.text
docs = loader.load()

# Chunk text content
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


# Index text content
def create_initial_chroma_db(
    ollama_model: str = "gemma:2b", directory: str = "chroma_db"
):
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model=ollama_model),
        persist_directory=directory,
    )
    return vectorstore


def get_existing_chroma_db(
    ollama_model: str = "gemma:2b", directory: str = "chroma_db"
) -> Chroma:
    return Chroma(
        embedding_function=OllamaEmbeddings(model=ollama_model),
        persist_directory=directory,
    )


# # Retrieve and generate using the relevant snippets of the blog.
prompt = "You are a running coach for amateur runners. You provide running plans for 1 week of running at a time. Runners can request plans in km or miles. 1 mile is 1.60934km. "
retriever = get_existing_chroma_db().as_retriever()


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ollama_llm(question, context):
    logging.info("Starting ollama up :D")
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(
        model="gemma:2b",
        messages=[{"role": "user", "content": formatted_prompt}],
        options=ollama.Options(temperature=0),
        stream=True,
    )
    if isinstance(response, Iterator):
        print_me: str = ""
        for idx, chunk in enumerate(response):
            print_me += chunk["message"]["content"]
            if idx % 5 == 0:
                # chunks are too small usually, so batch them before printing
                print(print_me)
                print_me = ""


def rag_chain(question):
    logging.info("Starting rag chain with question %s", question)
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)


# Use the RAG App
create_plan(30, 3)

# Currently creates wrong mileage every time. Hmm.
