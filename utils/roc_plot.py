import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, roc_curve, auc
import json
def roc_auc_result_logshow(label_values, predict_values, save_path, reverse=False):
    if reverse:
        pos_label = 0  
        print('AUC = {}'.format(1 - roc_auc_score(label_values, predict_values)))
    else:
        pos_label = 1
        print('AUC = {}'.format(roc_auc_score(label_values, predict_values)))

    
    fpr, tpr, thresholds = roc_curve(label_values, predict_values, pos_label=pos_label)

    
    roc_auc = auc(fpr, tpr)

    
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.loglog(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0.001, 1], [0.001, 1], 'r--')
    plt.xlim([0.001, 1.0])
    plt.ylim([0.001, 1.0])
    plt.ylabel('TPR')
    plt.xlabel('FPR')


    low = tpr[np.where(fpr<.001)[0][-1]]   

    
    f = interp1d(fpr, tpr, bounds_error=False, fill_value="extrapolate")

    
    fpr_001 = 0.001
    tpr_001 = float(f(fpr_001))

    fpr_01 = 0.01
    tpr_01 = float(f(fpr_01))

    
    if save_path:
        
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        
        plt.savefig(save_path + '.png', format='png', dpi=300, bbox_inches='tight')
        
        plt.savefig(save_path + '.pdf', format='pdf', dpi=300, bbox_inches='tight')

    
    json_save_path = os.path.join(os.path.dirname(save_path),
                                  f'tpr_at_0.001_and_0.01_FPR_{os.path.basename(save_path)}.json')
    with open(json_save_path, 'w') as f:
        json.dump({'tpr_at_0.001FPR': tpr_001, 'tpr_at_0.01FPR': tpr_01}, f)

    
    print(f'TPR at 0.001 FPR is {tpr_001}  ---interp1d')
    print(f'TPR at 0.001 FPR is {low}  ----- lira')
    

    
    plt.show()

def get_threshold_with_max_auc(label_values, predict_values):
    label_values = np.array(label_values)
    predict_values = np.array(predict_values)

    max_auc = 0  
    optimal_threshold = 0  

    
    for threshold in np.sort(np.unique(predict_values)):
        
        binary_predictions = (predict_values >= threshold).astype(int)
        
        current_auc = roc_auc_score(label_values, binary_predictions)

        
        if current_auc > max_auc:
            max_auc = current_auc
            optimal_threshold = threshold

    print("Max AUC:", max_auc)
    print("Optimal Threshold for max AUC:", optimal_threshold)

    
    return optimal_threshold



