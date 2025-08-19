import os
import random
import re
import pandas as pd
import json



def read_parquet(file_path):
    return pd.read_parquet(file_path)
def process_healthcare(): 
    
    file_path = './DatasetsRaw/HealthCareMagic/HealthCareMagic-100k-en.jsonl'
    file_name, extension = os.path.splitext(file_path)
    file_name = file_name.split("/")[3]
    
    docs = []
    with open(file_path, 'r') as file:
        
        for line in file:
            
            data = json.loads(line)
            docs.append(data["text"])
    
    docs_clean = []
    for text in docs:
        
        cleaned_text = re.sub(r"<(.*?)>", r"", text)
        doc = re.sub(r"(?m)^\s*(human:|bot:)", "", cleaned_text)
        doc = re.sub(r"^:", "", doc)
        doc = re.sub("\n", "\t", doc).strip()
        docs_clean.append(doc)
    random.shuffle(docs_clean)
    os.makedirs(f'./Datasets/HealthCareMagic/', exist_ok=True) 
    file_path_processed = "./Datasets/HealthCareMagic/" + file_name + "_processed" + ".json"
    with open(file_path_processed, 'w') as file:
        json.dump(docs_clean, file)

    print(f"HealthCareMagic finished.")
def process_agnews(): 
    final_data = []
    os.makedirs(f'./Datasets/AgNews/', exist_ok=True) 
    data = read_parquet(f"./DatasetsRaw/AgNews/train-00000-of-00001.parquet")
    for index, row in data.iterrows():
        text = row['text']
        final_data.append(text)
    with open("./Datasets/AgNews/ag_news_processed.json", 'w') as file:
        json.dump(final_data, file)

    print(f"AGNews finished.")
def process_nq():    
    final_data = []
    os.makedirs(f'./Datasets/NQ/', exist_ok=True) 
    for i in range(10):
        file_path = f'./DatasetsRaw/NQ/train-0000{i}-of-00024.parquet'
        data = read_parquet(file_path)
        for index, row in data.iterrows():
            question = row['question']
            answer = row['long']
            if len(answer)!=0:
                if 300 >= len(answer[0].split(" ")) >= 70:
                    final_data.append(question+"?"+"\t"+answer[0])
    for i in range(10, 24):
        file_path = f'./DatasetsRaw/NQ/train-000{i}-of-00024.parquet'
        data = read_parquet(file_path)
        for index, row in data.iterrows():
            question = row['question']
            answer = row['long']
            if len(answer)!=0:
                if 300 >= len(answer[0].split(" ")) >= 70:
                    final_data.append(question+"?"+"\t"+answer[0]) 
    with open("./Datasets/NQ/nq_processed.json", 'w') as file:
        json.dump(final_data, file)
    print(f"Natural Questions finished.")


def process_ms():  
    final_data = []
    os.makedirs(f'./Datasets/MsMarco/', exist_ok=True) 
    for i in range(7):
        file_path = f'./DatasetsRaw/MsMarco/train-0000{i}-of-00007.parquet'
        data = read_parquet(file_path)
        max_len = 0
        min_len = 1000
        for index, row in data.iterrows():
            if len(row['answers'][0].split(" "))  > max_len:
                max_len = len(row['answers'][0].split(" "))
            if len(row['answers'][0].split(" ")) < min_len:
                min_len = len(row['answers'][0].split(" "))

            if len(row['answers'])!=0 and len(row['answers'][0].split(" "))>=20:
                qa = row["query"] + "?\t" + row['answers'][0]
                
                final_data.append(qa)
        
        with open("./Datasets/MsMarco/ms_processed.json", 'w') as file:
            json.dump(final_data, file)
    print(f"Ms-Marco finished.")

if __name__ == '__main__':
    
    datasetName = "AGNews"    #HealthCareMagic   NQ    MsMarco  AGNews   

    if datasetName == "HealthCareMagic":
        process_healthcare()
    elif datasetName == "NQ":
        process_nq()
    elif datasetName == "MsMarco":
        process_ms()
    elif datasetName == "AGNews":
        process_agnews()
    else:
        print("Please check the name of the dataset.")
    
    



