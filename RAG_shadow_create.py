
from langchain_community.embeddings import HuggingFaceBgeEmbeddings 
from langchain_community.vectorstores import FAISS
from utils.file_read_utils import processed_data_read
from utils.vector_db_utils import create_model, save_samples


datasetName = "AGNews"    #HealthCareMagic   NQ    MsMarco  AGNews   

if datasetName == "HealthCareMagic":
    file_path = "./Datasets/HealthCareMagic/HealthCareMagic-100k-en_processed.json"
    file_name = "HealthCareMagic"
elif datasetName == "NQ":
    file_path = "./Datasets/NQ/nq_processed.json"
    file_name = "NQ"
elif datasetName == "MsMarco":
    file_path = "./Datasets/MsMarco/ms_processed.json"
    file_name = "MsMarco"
elif datasetName == "AGNews":
    file_path = "./Datasets/AgNews/ag_news_processed.json"
    file_name = "AgNews"
else:
    print("Please check the name of the dataset.")



model_name = "./EmbeddingModel/sentence-transformers/all_minilm_l6_v2"
model_kwargs = {'device': 'cuda'}  
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


data = processed_data_read(file_path)  


samples_member = data[10000:18000]
samples_nonmember = data[18000:19000]
sample_path = f'SamplesDatabase/{file_name}/ShadowModel'
index_folder_path = f"VectorDatabase/{file_name}/ShadowModel"


print(type(samples_member[0])) 



print(index_folder_path)
create_model(file_name, index_folder_path, samples_member, embeddings)    

save_samples("samples_member.json",sample_path,samples_member)  
save_samples("samples_nonmember.json",sample_path,samples_nonmember) 



if __name__=="__main__":
    
    print(file_name)
    vectordb = FAISS.load_local(index_folder_path, embeddings, file_name, allow_dangerous_deserialization=True)
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    response = retriever.invoke("Hi dr. samuelI am jawad, 24 years old, and for 4 years, i was sickthe sickness started from the point that i could not get decision about an issue that")
    
    for doc in response:
        print(doc)
        print("==============")