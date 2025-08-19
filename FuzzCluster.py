import json
import numpy as np
import torch
import torch.nn as nn
import Models as models  
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

import numpy as np
import skfuzzy as fuzz





def createFeatures_specified_tokennums(metrics, datasets_name, llm_name, token_nums):
    
    path_attack_result = f"AttackResult/{datasets_name}/{llm_name}_CSA_attack" 
    
    metricValuesFile = f'{path_attack_result}/PredictValues_Seq'
    
    
    
   
    numbers = token_nums

    allSequences = []
    number_of_samples = 0
    for i in numbers:   
       
        
        
        if "similarity" in metrics:
            
            similarityFile = f'{metricValuesFile}/predict_values_{i}.json'
            print(similarityFile)
            with open(similarityFile, 'r') as file:
                similarityValues = json.load(file)    
        else:
            similarityValues = []

        if "rouge1" in metrics:
            
            rouge1File = f'{metricValuesFile}/rouge1_{i}.json'
            print(rouge1File)
            with open(rouge1File, 'r') as file:
                rouge1Values = json.load(file)
        else:
            rouge1Values = []

        if "rouge2" in metrics:
            
            rouge2File = f'{metricValuesFile}/rouge2_{i}.json'
            print(rouge2File)
            with open(rouge2File, 'r') as file:
                rouge2Values = json.load(file)
        else:
            rouge2Values = []

        if "rougeL" in metrics:
            
            rougeLFile = f'{metricValuesFile}/rougeL_{i}.json'
            print(rougeLFile)
            with open(rougeLFile, 'r') as file:
                rougeLValues = json.load(file)
        else:
            rougeLValues = []

        if "rougeLsum" in metrics:
        
            rougeLsumFile = f'{metricValuesFile}/rougeLsum_{i}.json'
            print(rougeLsumFile)
            with open(rougeLsumFile, 'r') as file:
                rougeLsumValues = json.load(file)
        else:
            rougeLsumValues = []

        if "edit_distance" in  metrics: 
            edit_distanceFile = f'{metricValuesFile}/edit_distance_{i}.json'
            print(edit_distanceFile)
            with open(edit_distanceFile, 'r') as file:
                edit_distanceValues = json.load(file)
        else:
            edit_distanceValues = []

        
        if "RLEU4" in  metrics: 
            RLEU4File = f'{metricValuesFile}/RLEU4_{i}.json'
            print(RLEU4File)
            with open(RLEU4File, 'r') as file:
                RLEU4Values = json.load(file)
            RLEU4Values = [x / 100 for x in RLEU4Values]  
        else:
            RLEU4Values = []
        

    

        lists_to_stack = []
        
        for values in [similarityValues, rouge1Values, rouge2Values, rougeLValues, rougeLsumValues, edit_distanceValues, RLEU4Values]:
            if values:  
                lists_to_stack.append(values)
                
        
        if lists_to_stack:
            onePointofAllSamples = np.column_stack(lists_to_stack)
        else:
            print("no metric is used! exit!")
            exit()

        
        print(onePointofAllSamples.shape)  

        allSequences.append(onePointofAllSamples)

        
        number_of_samples = onePointofAllSamples.shape[0]
    
    print(len(allSequences))

    seq_data = []
    for i in range(0,number_of_samples):
        
        i_th_sample = []
        
        for j in range(0,len(numbers)):

            Seqs = allSequences[j] 

            j_th_point = Seqs[i,:]   


            i_th_sample.append(j_th_point)

        oneSeq = np.stack(i_th_sample, axis=0)   
        

        seq_data.append(oneSeq)


    
    #print(seq_data[0])
    #print(len(seq_data))       

    
    flattened_list = [arr.flatten(order='F') for arr in seq_data]

    
    #print(flattened_list[0])
    #print(flattened_list[0].shape)  

        


    return  flattened_list 


if __name__ == '__main__':

    
    llm_name = "meta"     # meta   mistralai   chatglm
    
    datasets_name = "HealthCareMagic"   #HealthCareMagic   NQ    MsMarco  AGNews 
    
    #token_nums = [10,30]
    token_nums = [10,30,50,70,90,110,130,150,170,190,210,230,250,270]
    
    #metrics = ["similarity", "rouge1", "rouge2", "rougeL", "rougeLsum", "edit_distance", "BLEU4"] 
    metrics = ["similarity", "rouge1"]   #Selection is required; however, using all available metrics can also achieve strong attack performance.

    if datasets_name == "AGNews":
        datasets_name = "AgNews"

    

    print("Used metrics:")
    print(metrics)

    target_data = createFeatures_specified_tokennums(metrics, datasets_name, llm_name, token_nums)

    X = np.stack(target_data)  

    
    true_labels = np.array([1]*1000 + [0]*1000)


    
    num_clusters = 2

    
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, num_clusters, 2, error=0.005, maxiter=1000, init=None)

    
    membership_first_cluster = u[0]


    
    myauc_temp = roc_auc_score(true_labels, membership_first_cluster)
    if myauc_temp > 0.5:
        membership_first_cluster = u[0]
    else:
        membership_first_cluster = u[1]
    
    
    myauc = roc_auc_score(true_labels, membership_first_cluster)

    print(f"AUC: {myauc:.4f}")

    
    fpr, tpr, thresholds = roc_curve(true_labels, membership_first_cluster)

    low = tpr[np.where(fpr<.001)[0][-1]]   
    print(f'TPR at 0.001 FPR is {low}')


    pre_labels = np.where(membership_first_cluster >= 0.5, 1, 0) 
    acc = accuracy_score(true_labels, pre_labels)
    print(f"Accuracy (ACC): {acc:.4f}")

    





