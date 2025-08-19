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

def ROC_AUC_Result_logshow(label_values,predict_values,reverse=False):
    if reverse:
        pos_label = 0   
        print('AUC = {}'.format(1-roc_auc_score(label_values,predict_values)))
    else:
        pos_label = 1
        print('AUC = {}'.format(roc_auc_score(label_values,predict_values)))
    fpr,tpr,thresholds = roc_curve(label_values,predict_values,pos_label=pos_label)  
    
    roc_auc = auc(fpr,tpr)
    plt.title('Receiver Operating Characteristic(ROC)')
    
    plt.loglog(fpr,tpr,'b', label='AUC=%0.4f' %roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0.001,1],[0.001,1],'r--')
    plt.xlim([0.001,1.0])
    plt.ylim([0.001,1.0])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    
    low = tpr[np.where(fpr<.001)[0][-1]]  
    print('TPR at 0.001 FPR is {}'.format(low))


    plt.cla()



def createSequences_specified_tokennums(metrics, datasets_name, llm_name, target_shadow_Flag = "target", token_nums=[]):
    
    
    
    path_attack_result = f"AttackResult/{datasets_name}/{llm_name}_CSA_attack" 
    if target_shadow_Flag == "target":
        metricValuesFile = f'{path_attack_result}/PredictValues_Seq'
    else:
        metricValuesFile = f'{path_attack_result}/PredictValues_Seq_Shadow'
    
    
    
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

        
        if "BLEU4" in  metrics: 
            BLEU4File = f'{metricValuesFile}/BLEU4_{i}.json'
            print(BLEU4File)
            with open(BLEU4File, 'r') as file:
                BLEU4Values = json.load(file)
            BLEU4Values = [x / 100 for x in BLEU4Values]  
        else:
            BLEU4Values = []
        

        
        

        lists_to_stack = []
        
        for values in [similarityValues, rouge1Values, rouge2Values, rougeLValues, rougeLsumValues, edit_distanceValues, BLEU4Values]:
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

            if i < number_of_samples/2:   
                labeled_j_th_point = np.insert(j_th_point, 0, 1)   
            else:     
                labeled_j_th_point = np.insert(j_th_point, 0, 0)


            i_th_sample.append(labeled_j_th_point)

        oneSeq = np.stack(i_th_sample, axis=0)   
        


            

        seq_data.append(torch.Tensor(oneSeq))


    return seq_data    



def train_attack_model_RNN(dataset,epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=50, model='rnn'): 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_x, test_x = dataset
    
    num_classes = 2

    
    if batch_size > len(train_x):
        batch_size = len(train_x)

    
    print('Building model with {} training data, {} classes...'.format(len(train_x), num_classes))

     
    train_data = models.TrData(train_x)  
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                             collate_fn=models.collate_fn)

   
    test_data = models.TrData(test_x)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             collate_fn=models.collate_fn)


    onetr = train_data[0] 
    onepoint_size = onetr.size(1) 
    
    input_size = onepoint_size -1
    hidden_size = 50 
    num_layers = 1 

    
    if model == 'rnn':
        print('Using an RNN based model for attack...')
        net = models.lstm(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,num_classes=num_classes,batch_size=batch_size)
        net = net.to(device)
    elif model == 'rnnAttention':
        print('Using an RNN with atention model for attack...')
        net = models.LSTM_Attention(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,num_classes=num_classes,batch_size=batch_size)
        net = net.to(device)
    else:
        print('Using an error type for attack model...')

    
    net.train()
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    
    
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
              {'params': no_decay_list, 'weight_decay': 0.}]
    
    learning_rate = learning_rate
    
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=l2_ratio)  
    
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9) 

    
    print('model: {},  device: {},  epoch: {},  batch_size: {},   learning_rate: {},  l2_ratio: {}'.format(model, device, epochs, batch_size, learning_rate, l2_ratio))
    count = 1
    print('Training...')
    for epoch in range(epochs):
        running_loss = 0


        for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(train_loader):	
            

            X_vector = X_vector.to(device)  
            Y_vector = Y_vector.to(device)  

            output,_ = net(X_vector,len_of_oneTr)
            

            output = output.squeeze(0)  

            
            Y_vector = Y_vector.long()

            loss = criterion(output, Y_vector)    
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        if optimizer.param_groups[0]['lr']>0.0005:  
            scheduler.step()  

        if (epoch + 1) % 10 == 0: 
            
            print('Epoch: {}, Loss: {:.5f},  lr: {}'.format(epoch + 1, running_loss, optimizer.param_groups[0]['lr']))


    print("Training finished!")

    
    print('Testing...')
    pred_y = []
    pred_y_prob = []
    test_y = []

    hidden_outputs = []
    net.eval()  

    
    if batch_size > len(test_x):
        batch_size = len(test_x)
    for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(test_loader):	
        

        X_vector = X_vector.to(device)
        Y_vector = Y_vector.to(device)

        output, hidden_output = net(X_vector,len_of_oneTr)
        output = output.squeeze(0)  
        out_y = output.detach().cpu() 
        pred_y.append(np.argmax(out_y, axis=1))  
        pred_y_prob.append(out_y[:, 1]) 
        test_y.append(Y_vector.detach().cpu())

        hidden_output = hidden_output.detach().cpu()
        hidden_output = np.squeeze(hidden_output)
        hidden_outputs.append(hidden_output)   


       


    pred_y = np.concatenate(pred_y)  
    pred_y_prob = np.concatenate(pred_y_prob)

    hidden_outputs = np.concatenate(hidden_outputs)

    test_y = np.concatenate(test_y)  
    print('Accuracy: {}'.format(accuracy_score(test_y, pred_y)))  
    
    
    ROC_AUC_Result_logshow(test_y,pred_y_prob,reverse=False)

    return test_y,pred_y_prob, hidden_outputs




        
def AttackingWithShadowTraining_RNN(X_train, X_test, epochs=50,batch_size=20, modelType='rnn'):
    dataset = (X_train,
               X_test)
    l2_ratio = 0.0001   
    targetY, pre_member_label, hidden_outputs = train_attack_model_RNN(dataset=dataset,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    learning_rate=0.01,
                                    n_hidden=64,
                                    l2_ratio = l2_ratio,
                                    model= modelType)

    return targetY, pre_member_label, hidden_outputs






if __name__ == '__main__':

    
    
    llm_name = "meta"     # meta   mistralai   chatglm
    
    datasets_name = "HealthCareMagic"   #HealthCareMagic   NQ    MsMarco  AGNews 
    
    #token_nums = [10,30]
    token_nums = [10,30,50,70,90,110,130,150,170,190,210,230,250,270]
    
    metrics = ["similarity", "rouge1", "rouge2", "rougeL", "rougeLsum", "edit_distance", "BLEU4"] 
    #metrics = ["similarity"]

    attack_epoch = 200  

    
    print("Used metrics:")
    print(metrics)

    if datasets_name == "AGNews":
        datasets_name = "AgNews"

    
    target_shadow_Flag = "target"   
    
    target_seq_data = createSequences_specified_tokennums(metrics, datasets_name, llm_name, target_shadow_Flag, token_nums)

    
    target_shadow_Flag = "shadow"    
    
    shadow_seq_data = createSequences_specified_tokennums(metrics, datasets_name, llm_name, target_shadow_Flag, token_nums)


    print("training attack model...")

    
        
    
    modelType = 'rnnAttention' 

    targetY, pre_member_label, hidden_outputs = AttackingWithShadowTraining_RNN(shadow_seq_data, target_seq_data, epochs=attack_epoch, batch_size=100, modelType = modelType )  

