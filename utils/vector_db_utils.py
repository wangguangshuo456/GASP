import json
import os

from langchain_community.embeddings import HuggingFaceBgeEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def create_model(file_name,index_folder_path,samples,embeddings):

    
    documents = [Document(page_content=text) for text in samples]
    print(len(documents))
    vectordb = FAISS.from_documents(documents, embeddings)
    index_name = file_name  
    print(index_name)
    
    vectordb.save_local(index_folder_path, index_name)    


def save_samples(file_name,path_sample,samples):

    
    os.makedirs(path_sample, exist_ok=True)  
    file_path = os.path.join(path_sample, file_name)
    with open(file_path, 'w') as file:
        json.dump(samples, file)         


def get_retriever(index_folder_path,index_name,top_k):
    
    model_name = "./EmbeddingModel/sentence-transformers/all_minilm_l6_v2"    
    model_kwargs = {'device': 'cuda'}  
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    
    vectordb = FAISS.load_local(index_folder_path, embeddings, index_name, allow_dangerous_deserialization=True)
    
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    return retriever