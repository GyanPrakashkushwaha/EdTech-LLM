from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from logging_and_exception import CustomException

def loadRetriever(embeddingFolder:str = 'embeddingTransformers',dbFolder:str ='DB'):
    model_name = "hkunlp/instructor-large"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceInstructEmbeddings(
        cache_folder = embeddingFolder,
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    vectorDB = FAISS.load_local(dbFolder,embeddings=hf)
    
    return vectorDB

