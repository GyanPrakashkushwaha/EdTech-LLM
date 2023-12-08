from secret.secret import apiKey
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings # Embeddings
from langchain.vectorstores import FAISS # for VectorDB
from logging_and_exception import logger , CustomException
import os

chat = GooglePalm(google_api_key=apiKey,temperature=0.8,verbose=True)

response = chat("""Act as a PPT creator for fees structure for IITM Bs Degree students.
                create a professional powerpoint presentation please show the complete fees in different slides
                for all categories because the main aim is to show the fees of IITM""")

# load data
loader = CSVLoader(file_path=r'DB/data.csv',source_column='prompt')
data = loader.load()
# print(data)
# print(response)

def embed():
    os.makedirs(name='embeddingTransformers',exist_ok=True)
    model_name = "hkunlp/instructor-large"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceInstructEmbeddings(
        cache_folder = r'embeddingTransformers',
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    vectorDB = FAISS.from_documents(documents=data,embedding=hf)
    vectorDB.save_local(r'DB')

    return (
        hf,vectorDB
    )
    

    
try:
    embed()
except Exception as e:
    raise CustomException(e)

