from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import Binarizer

def get_model_eval(actual,pred):
    con = confusion_matrix(actual,pred)
    acc = accuracy_score(actual,pred)
    pre = precision_score(actual,pred)
    rec = recall_score(actual,pred)
    f1 = f1_score(actual,pred)
    print("오차 행렬\n",con)
    print(f"정확도:{acc:6.4f} 정밀도:{pre:6.4f}")
    print(f"재현율:{rec:6.4f} F1 점수:{f1:6.4f}")
    
def get_eval_by_thresholds(y_test,pred,thresholds):
    for threshold in thresholds:
        print("=== threshold:",threshold)
        bn = Binarizer(threshold=threshold)
        pv = bn.fit_transform(pred)
        get_model_eval(y_test,pv)