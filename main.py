## Integrate our code OpenAI API
import os
from constants import groq_key
#from langchain_community.llms import OpenAI

import streamlit as st #df is your dataframe

from langchain_groq import ChatGroq


# Stocke ta clé API Groq
os.environ["GROQ_API_KEY"] = groq_key

# Crée le modèle (ici LLaMA 3 8B)
llm = ChatGroq(
    model="llama3-8b-8192",  # ou "mixtral-8x7b-32768" si tu veux plus puissant
    temperature=1
)

# Test
st.title("Langchain Demo with Groq API")
input_text = st.text_input("Enter your query")


if input_text:
    response = llm.invoke(input_text)
    st.write( response.content)

## OPENAI LLMS
#llm = OpenAI(temperature=0.9, openai_api_key=openai_key)

