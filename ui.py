import streamlit as st
from main import chatLLM



st.title('👀LLM🌳🌴')
question = st.text_area('Question')
go = st.button('Go🚀')

if question:
    if go:
        chat = chatLLM()
        response = chat(question)
        st.header('answer')
        st.write(response['result'])
    
    
