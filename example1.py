## Integrate our code OpenAI API
import os
from constants import groq_key
#from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st #df is your dataframe

from langchain_groq import ChatGroq


# Stocke ta clé API Groq
os.environ["GROQ_API_KEY"] = groq_key

# Test
st.title("Celebrity Search Results")
input_text = st.text_input("Enter your query")


# Prompt Template
first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about {name}"
)

# Memory
person_memory = ConversationBufferMemory(
    input_key="name",
    memory_key="chat_history"
)
dob_memory = ConversationBufferMemory(
    input_key="person",
   memory_key="chat_history")
description_memory = ConversationBufferMemory(
    input_key="dob",
    memory_key="description_history"
)
llm = ChatGroq(
    model="llama3-8b-8192",  # ou "mixtral-8x7b-32768" si tu veux plus puissant
    temperature=0.3
)

chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose =True, output_key="person", memory=person_memory)
# Crée le modèle (ici LLaMA 3 8B)
 
 # Prompt Template
second_input_prompt = PromptTemplate(
    input_variables=["person"],
    template="When was {person} born"
)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose =True, output_key="dob", memory=dob_memory)   

 # Prompt Template
third_input_prompt = PromptTemplate(
    input_variables=["dob"],
    template="Mention 5 major events hapened around {dob} in the world"
)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose =True, output_key="description", memory=description_memory)   

 # Prompt Template
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3], input_variables=["name"], output_variables=["person", "dob", "description"],
    verbose=True
) 
if input_text:
    response = llm.invoke(input_text)
    st.write( parent_chain({"name":response.content}) )
    with st.expander("Person name"):
        st.info(person_memory.buffer)
    with st.expander("Major Events"):
        st.info(description_memory.buffer)
