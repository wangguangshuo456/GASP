import json
import os
from utils.file_read_utils import processed_data_read

import sacrebleu

def calculate_bleu(target, inference):
    
    
    references = [[target]]  
    hypothesis = inference    

    
    bleu_score = sacrebleu.corpus_bleu([hypothesis], references)

    
    return bleu_score.score

if __name__ == '__main__':

    
    llm_name = "meta"     # meta   mistralai   chatglm

    datasets_name = "HealthCareMagic"   #HealthCareMagic   NQ    MsMarco  AGNews  
    
    max_tokens_list = [10,30,50,70,90,110,130,150,170,190,210,230,250,270]
    #max_tokens_list = [10,30]

    numbers = max_tokens_list

    if datasets_name == "AGNews":
        datasets_name = "AgNews"
    

    
    samples_path = f"SamplesDatabase/{datasets_name}/TargetModel/"
    samples_mem = processed_data_read(samples_path + "samples_member.json")[:1000]  
    samples_nonmem = processed_data_read(samples_path + "samples_nonmember.json")[:1000]  

    
    path_attack_result = f"AttackResult/{datasets_name}/{llm_name}_CSA_attack"

    metricValuesFile = f'{path_attack_result}/PredictValues_Seq'

    


    print(numbers)

    all_data = []
    for i in numbers:  

       
        path_LLM_mem = f"{metricValuesFile}/LLMMemberOutputs_{i}.json"
        path_LLM_nonmem = f"{metricValuesFile}/LLMNonMemberOutputs_{i}.json"

        print(path_LLM_mem)
        print(path_LLM_nonmem)

        LLM_outputs_mem = processed_data_read(path_LLM_mem)
        LLM_outputs_nonmem = processed_data_read(path_LLM_nonmem)

       


        
        BLEU4_mem = []
        for prediction, reference in zip(LLM_outputs_mem, samples_mem):
            score = calculate_bleu(reference,prediction)
            BLEU4_mem.append(score)

       
        BLEU4_nonmem = []
        for prediction, reference in zip(LLM_outputs_nonmem, samples_nonmem):
            score = calculate_bleu(reference, prediction)
            BLEU4_nonmem.append(score)

        BLEU4_CSA = BLEU4_mem + BLEU4_nonmem
        with open(f"{metricValuesFile}/BLEU4_{i}.json", 'w') as file:
            json.dump(BLEU4_CSA, file)  