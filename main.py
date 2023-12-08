from retrieve import loadRetriever
from logging_and_exception import CustomException
from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm
# from secret.secret import G_PALM_API as apiKeyGPalm
from dotenv import load_dotenv
import os
load_dotenv()

apiKeyGPalm = os.environ['G_PALM_API']

def chatLLM():
    gPalm = GooglePalm(google_api_key=apiKeyGPalm,temperature=0.8,verbose=True)
    vectorDB = loadRetriever()
    retriever = vectorDB.as_retriever()
    # obj = retriever.get_relevant_documents('Is this bootcamp enough for me in Microsoft Power BI and Excel certifications?')
    
    chat = RetrievalQA.from_chain_type(llm = gPalm,
                retriever= retriever,
                input_key='query',
                return_source_documents=True)  
    
    return chat
    

# try:
#     gPalm = GooglePalm(google_api_key=apiKeyGPalm,temperature=0.8,verbose=True)
#     vectorDB = loadRetriever()
#     retriever = vectorDB.as_retriever()
#     obj = retriever.get_relevant_documents('Is this bootcamp enough for me in Microsoft Power BI and Excel certifications?')
    
#     chat = RetrievalQA.from_chain_type(llm = gPalm,
#                 retriever= retriever,
#                 input_key='query',
#                 return_source_documents=True)    
    
#     print(chat('Hello'))

# except Exception as e:
#     CustomException(e)
    