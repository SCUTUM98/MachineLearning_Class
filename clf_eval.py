from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score

# 정밀도와 재현율
def get_clf_eval(actual, pred=None, pred_proba=None):
    con = confusion_matrix(actual, pred)
    acc = accuracy_score(actual, pred)
    pre = precision_score(actual, pred)
    rec = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred_proba)
    
    print('오차 행렬\n', con)
    print(f'정확도;{acc:6.4f} 정밀도:{pre:6.4f} 재현율:{rec:6.4f} F1:{f1:6.4f}')
    print(f'ROC AUC:{roc_auc:6.4f}')