# app.py

import os
from dotenv import load_dotenv
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter
import streamlit as st

# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_PROJECT'] = "Multi_query_rag"

# Streamlit app starts here
st.title("Multi-Query UFC RAG 🤼")

# Prompt to generate multi-perspective queries
multi_query_template = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question} 
"""
prompt_perspectives = ChatPromptTemplate.from_template(multi_query_template)

# Final RAG prompt
rag_template = """Answer the following question based on this context:

{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

@st.cache_resource
def setup_chains():
    loader = WebBaseLoader(
        web_paths=('https://en.wikipedia.org/wiki/Ultimate_Fighting_Championship',),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer('body'))
    )
    blog_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(blog_docs)

    vectorstorage = Chroma.from_documents(splits, embedding=MistralAIEmbeddings(), persist_directory='chroma_db')
    retriever = vectorstorage.as_retriever()
    vectorstorage.persist()

    llm = ChatMistralAI(temperature=0, model_name='mistral-large-latest')

    generate_queries = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    def get_unique_union(documents: list[list]):
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    # Multi-query retriever chain
    retrieval_chain = generate_queries | retriever.map() | get_unique_union

    # Final RAG chain: retrieve -> prompt -> LLM -> output
    final_rag_chain = (
        {
            "context": retrieval_chain,
            "question": itemgetter("question")
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return final_rag_chain

# Initialize once
final_rag_chain = setup_chains()

# Input
question = st.text_input("Ask your UFC-related question:")

if question:
    with st.spinner("Retrieving and answering..."):
        answer = final_rag_chain.invoke({"question": question})
        st.success("Answer generated!")
        st.markdown(f"### 🤖 Answer:\n{answer}")
