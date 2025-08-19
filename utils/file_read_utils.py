import json
import os

def original_data_read(file_path):
    
    health_data = []
    with open(file_path, 'r') as file:
        
        for line in file:
            
            data = json.loads(line)
            health_data.append(data["text"])

    return health_data




def processed_data_read(file_path):
    
    with open(file_path, 'r') as file:
        file_list = json.load(file)
    
    return file_list

