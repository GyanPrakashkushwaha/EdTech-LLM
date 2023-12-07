from secret.secret import apiKey
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings # Embeddings
from langchain.vectorstores import FAISS # for VectorDB
from logging_and_exception import logger , CustomException

chat = GooglePalm(google_api_key=apiKey,temperature=0.8,verbose=True)

response = chat("""Act as a PPT creator for fees structure for IITM Bs Degree students.
                create a professional powerpoint presentation please show the complete fees in different slides
                for all categories because the main aim is to show the fees of IITM""")

# load data
loader = CSVLoader(file_path=r'DB/data.csv',source_column='prompt')
data = loader.load()
# print(data)
# print(response)

try:
    embeddings = HuggingFaceInstructEmbeddings(
        query_instruction="Represent the query for retrieval: "
    )

except Exception as e:
    raise CustomException(e)