import langchain.docstore
import streamlit as st

from langchain.llms import HuggingFaceHub
api_key="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"

import langchain
langchain.debug=True

import pickle
vector_store = pickle.load(open('vectorstore.pkl', 'rb'))

llm = HuggingFaceHub(
    huggingfacehub_api_token=api_key,
    repo_id="google/flan-t5-large",
    model_kwargs={
        "temperature": 0.5,
        "top_p": 0.85,
        "max_length": 150  # Increase max_length for longer outputs
    }
)

st.title("Chatbot for Codebasics")

# Get user input
user_input = st.text_input("Enter your question:")

if user_input:
    context = vector_store.similarity_search(user_input, k=2)[0].page_content
    answer = llm(f'Answer the question: "{user_input}" based on the context only: "{context}"\n If the answer is not in the context, say "I don\'t know". Do not make things up')
    st.write(answer)