import json
import os
from utils.file_read_utils import processed_data_read
import Levenshtein



def calculate_edit_distance(target, inference):
    
    ld = Levenshtein.distance(target, inference)

    
    max_len = max(len(target), len(inference))

    
    if max_len == 0:
        return 1.0  

    
    nls = 1 - (ld / max_len)
    return nls

if __name__ == '__main__':

    
    llm_name = "meta"     # meta   mistralai   chatglm

    datasets_name = "HealthCareMagic"   #HealthCareMagic   NQ    MsMarco  AGNews  
    
    max_tokens_list = [10,30,50,70,90,110,130,150,170,190,210,230,250,270]
    #max_tokens_list = [10,30]

    numbers = max_tokens_list

    if datasets_name == "AGNews":
        datasets_name = "AgNews"
    
    samples_path = f"SamplesDatabase/{datasets_name}/ShadowModel/"
    samples_mem = processed_data_read(samples_path + "samples_member.json")[:1000]  
    samples_nonmem = processed_data_read(samples_path + "samples_nonmember.json")[:1000]  

    
    path_attack_result = f"AttackResult/{datasets_name}/{llm_name}_CSA_attack"

    metricValuesFile = f'{path_attack_result}/PredictValues_Seq_Shadow'

    
    print(numbers)

    all_data = []
    for i in numbers:  

        
        path_LLM_mem = f"{metricValuesFile}/LLMMemberOutputs_{i}.json"
        path_LLM_nonmem = f"{metricValuesFile}/LLMNonMemberOutputs_{i}.json"

        print(path_LLM_mem)
        print(path_LLM_nonmem)

        LLM_outputs_mem = processed_data_read(path_LLM_mem)
        LLM_outputs_nonmem = processed_data_read(path_LLM_nonmem)

        


        
        edit_distance_mem = []

        for prediction, reference in zip(LLM_outputs_mem, samples_mem):
            distance = calculate_edit_distance(reference,prediction)
            edit_distance_mem.append(distance)

        
        edit_distance_nonmem = []

        for prediction, reference in zip(LLM_outputs_nonmem, samples_nonmem):
            distance = calculate_edit_distance(reference,prediction)
            edit_distance_nonmem.append(distance)

        edit_distance = edit_distance_mem + edit_distance_nonmem
        

        if not os.path.exists(metricValuesFile):
            os.makedirs(metricValuesFile)
        with open(f"{metricValuesFile}/edit_distance_{i}.json", 'w') as file:
            json.dump(edit_distance, file)  
