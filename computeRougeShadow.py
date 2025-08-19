
import json
import numpy as np


from utils.file_read_utils import processed_data_read
from rouge_score import rouge_scorer


if __name__ == '__main__':

    
    llm_name = "meta"     # meta   mistralai   chatglm

    datasets_name = "HealthCareMagic"   #HealthCareMagic   NQ    MsMarco  AGNews  
    
    max_tokens_list = [10,30,50,70,90,110,130,150,170,190,210,230,250,270]
    #max_tokens_list = [10,30]

    numbers = max_tokens_list

    if datasets_name == "AGNews":
        datasets_name = "AgNews"
    
    samples_path = f"SamplesDatabase/{datasets_name}/ShadowModel/"   
    samples_mem = processed_data_read(samples_path+"samples_member.json")[:1000]     
    samples_nonmem = processed_data_read(samples_path+"samples_nonmember.json")[:1000]    
    
    
    
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


        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        
        
        
        rouge1_mem = []
        rouge2_mem = []
        rougeL_mem = []
        rougeLsum_mem = []    
        for prediction, reference in zip(LLM_outputs_mem,samples_mem):
            score = scorer.score(prediction, reference)
            rouge1_mem.append(score['rouge1'].fmeasure)
            rouge2_mem.append(score['rouge2'].fmeasure)
            rougeL_mem.append(score['rougeL'].fmeasure)
            rougeLsum_mem.append(score['rougeLsum'].fmeasure)

        
        rouge1_nonmem = []
        rouge2_nonmem = []
        rougeL_nonmem = []
        rougeLsum_nonmem = []       
        for prediction, reference in zip(LLM_outputs_nonmem,samples_nonmem):
            score = scorer.score(prediction,reference)
            rouge1_nonmem.append(score['rouge1'].fmeasure)
            rouge2_nonmem.append(score['rouge2'].fmeasure)
            rougeL_nonmem.append(score['rougeL'].fmeasure)
            rougeLsum_nonmem.append(score['rougeLsum'].fmeasure)

        rouge1_CSA = rouge1_mem + rouge1_nonmem
        
        with open(f"{metricValuesFile}/rouge1_{i}.json", 'w') as file:
            json.dump(rouge1_CSA,file)   

        rouge2_CSA = rouge2_mem + rouge2_nonmem
        with open(f"{metricValuesFile}/rouge2_{i}.json", 'w') as file:
            json.dump(rouge2_CSA,file)   

        rougeL_CSA = rougeL_mem + rougeL_nonmem
        with open(f"{metricValuesFile}/rougeL_{i}.json", 'w') as file:
            json.dump(rougeL_CSA,file)   


        rougeLsum_CSA = rougeLsum_mem + rougeLsum_nonmem
        with open(f"{metricValuesFile}/rougeLsum_{i}.json", 'w') as file:
            json.dump(rougeLsum_CSA,file)   