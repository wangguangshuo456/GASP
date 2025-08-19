from utils.file_read_utils import processed_data_read
from utils.vector_db_utils import get_retriever
from utils.roc_plot import roc_auc_result_logshow
from utils.similarity_get import get_semantic_similarity
from utils.llms import RagLLM1, RagLLM2,RagLLM3
import re
import json
import os


llm_name = "meta"     # meta   mistralai   chatglm

datasets_name = "HealthCareMagic"   #HealthCareMagic   NQ    MsMarco  AGNews   

#max_tokens_list = [270]
max_tokens_list = [10,30,50,70,90,110,130,150,170,190,210,230,250,270]

########################################################


if llm_name == "meta":  
    llm = RagLLM1()
elif llm_name == "mistralai":
    llm = RagLLM2()
elif llm_name == "chatglm":
    llm = RagLLM3()


if datasets_name == "AGNews":
    datasets_name = "AgNews"

for max_tokens in max_tokens_list:
    
    split_signals = "\t"   
    
    path_attack_result = f"AttackResult/{datasets_name}/{llm_name}_CSA_attack"    

    index_folder_path = f"VectorDatabase/{datasets_name}/TargetModel"
    index_name = datasets_name
    
    top_k = 4
    Retriever = get_retriever(index_folder_path, index_name, top_k)  

    def get_prompt_with_db(user_prompt,retriever):
        
        retrieved_documents = [doc.page_content for doc in retriever.invoke(user_prompt)]
        
        for i in range(len(retrieved_documents)):
            retrieved_documents[i] = "\"" + retrieved_documents[i] + "\""
        context = '\n'.join(retrieved_documents)
        
        sys_prompt = f'''Please answer the question using the Context provided.
                    Context: {context}
                    Question: {user_prompt}'''
        
        return sys_prompt

    
    def mqa_attack(samples, mem_flag) :  
        result_list = []
        llm_results = []

        for i, sample in enumerate(samples):
            

            if datasets_name == "AGNews":
                text = " ".join(samples[i].split(" ")[:10]) 
                input_string = f"Complete this sentence \"{text}......\" based on the Context."  
                prompt = get_prompt_with_db(input_string,Retriever)  

            else:
                text = samples[i].split(split_signals)[0]  
                input_string = text  
                prompt = get_prompt_with_db(input_string,Retriever)   
            

            
            result= llm.invoke(prompt, max_tokens).strip()
            
            llm_results.append(result)
            result = re.sub("\n"," ", result)  
            
            similarity = get_semantic_similarity(result, sample)  
            print("[",similarity,"]")
            result_list.append(similarity)
            print(f"Attack {i} finished.")

        return result_list,llm_results





    
    path_members = f"SamplesDatabase/{datasets_name}/TargetModel/samples_member.json"
    path_nonmembers = f"SamplesDatabase/{datasets_name}/TargetModel/samples_nonmember.json"
    samples_member = processed_data_read(path_members)
    samples_nonmember = processed_data_read(path_nonmembers)
    print(len(samples_member),len(samples_nonmember))
    print(len(samples_member), len(samples_nonmember))


    
    mem_mqa_results,llm_results_member= mqa_attack(samples_member[:1000], True)  
    
    none_mem_mqa_results, llm_results_nonmember = mqa_attack(samples_nonmember, False)   
   

    os.makedirs(f'{path_attack_result}/PredictValues_Seq/', exist_ok=True)  


    
    predict_values = mem_mqa_results + none_mem_mqa_results   
    with open(f'{path_attack_result}/PredictValues_Seq/predict_values_{max_tokens}.json', 'w') as f:
        json.dump(predict_values, f)
    
    label_values = [1] * 1000 + [0] * 1000     
    
    with open(f'{path_attack_result}/PredictValues_Seq/label_values_{max_tokens}.json', 'w') as f:
        json.dump(label_values, f)



    
    with open(f'{path_attack_result}/PredictValues_Seq/LLMMemberOutputs_{max_tokens}.json', 'w') as f:   
        json.dump(llm_results_member, f)
    with open(f'{path_attack_result}/PredictValues_Seq/LLMNonMemberOutputs_{max_tokens}.json', 'w') as f:   
        json.dump(llm_results_nonmember, f)
    


    os.makedirs(f'{path_attack_result}/AUC_ROC_Seq/roc', exist_ok=True)  

    roc_save_path = f'{path_attack_result}/AUC_ROC_Seq/roc'
    
