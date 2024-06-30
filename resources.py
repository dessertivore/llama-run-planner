from typing import Iterator
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from bs4 import BeautifulSoup, SoupStrainer

import requests

# from langchain import hub
# from langchain_chroma import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_text_splitters import RecursiveCharacterTextSplitter


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

# Load, chunk and index the contents of the blog.
html_doc = requests.get(
    "https://www.runnersloveyoga.com/new-blog/2021/2/9/how-to-create-your-own-training-plan-with-a-printable-pdf"
)
soup = BeautifulSoup(
    html_doc.text,
    "html.parser",
    parse_only=SoupStrainer("p", attrs={"style": "white-space:pre-wrap;"}),
)
soup_str = str(soup)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_text(soup_str)
print(splits)
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vectorstore.as_retriever()
# prompt = hub.pull("rlm/rag-prompt")


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# rag_chain.invoke("What is Task Decomposition?")
