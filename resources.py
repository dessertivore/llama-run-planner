from typing import Iterator
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.llms.ollama import Ollama

import requests
from langchain_community.embeddings import OllamaEmbeddings

from langchain import hub
from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

my_llm = Ollama(
    model="gemma:2b",
    temperature=0.1,
)


def create_plan(weekly_kilometres: int, weekly_runs: int):
    stream = ollama.chat(
        model="gemma:2b",
        options=ollama.Options(temperature=0.1, mirostat=2),
        messages=[
            {
                "role": "system",
                "content": "You are a running coach for amateur runners. Your running plans contain at least 1 tempo or interval run and 1 long run (distance for this should be 50%\ or less of weekly mileage).",
                "role": "user",
                "content": f"Please write me a running plan. I want to run exactly {weekly_kilometres}km per week, spread out over {weekly_runs} runs. Format should strictly be 'Day 1- run type, distance (km). Day 2- run type, distance (km), etc. You must not go over weekly mileage specified.",
            }
        ],
        stream=True,
    )
    if isinstance(stream, Iterator):
        for chunk in stream:
            print(chunk["message"]["content"])


# (create_plan(40, 3))

# Load content of blog
html_doc = requests.get(
    "https://run.outsideonline.com/training/rules-when-setting-up-weekly-running-schedule/"
)
soup = BeautifulSoup(
    html_doc.text,
    "html.parser",
    parse_only=SoupStrainer(
        "article",
        attrs={"class": "l-container l-article l-article--post u-spacing--triple"},
    ),
)

# Get rid of tags
soup = soup.text.replace("\n", " ")

# Chunk text content
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_text(soup)

# Index text content
vectorstore = Chroma.from_texts(
    texts=splits, embedding=OllamaEmbeddings(model="gemma:2b")
)

# # Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = "You are a running coach for amateur runners. Your running plans contain at least 1 tempo or interval run and 1 long run (distance for this should be 50%\ or less of weekly mileage)."


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | my_llm
    | StrOutputParser()
)
for chunk in rag_chain.stream("Please give me a 40mpw running plan?"):
    print(chunk, end="", flush=True)
print(rag_chain.invoke("Please give me a 40mpw running plan?"))
