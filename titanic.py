from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import Binarizer
import numpy as np
import matplotlib.pyplot as plt

# 입력된 나이에 따른 구분 값을 반환하는 함수
def get_category(age):
    cate = ""
    if age<0:
        cate = 'Unknown'
    elif age <= 5:
        cate = "BABY"
    elif age <= 12:
        cate = "CHILD"
    elif age <= 18:
        cate = "TEEN"
    elif age <= 25:
        cate = "STUDENT"
    elif age <= 35:
        cate = "YOUNG"
    elif age <= 60:
        cate = "ADULT"
    else:
        cate = "ELDERLY"
    return cate

def encode_features(df,columns):
    for column in columns:
        le = LabelEncoder()
        le.fit(df[column])
        df[column] = le.transform(df[column])
        
    return df

names_values = ['Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'ETC']
def get_name_index(name):
    first, second = name.split(',')
    foos = second.split('.')
    tn = foos[0].replace(' ', '')
    
    for i in range(5):
        if tn == names_values[i]:
            return i
    return 5

## 교차검증 함수
def exec_kfold(clf, xdf, ydf, folds = 5):
    kfold = KFold(n_splits=folds)
    scores = []
    
    for iter_count, (train_index, test_index) in enumerate(kfold.split(xdf)):
        x_train, x_test = xdf.values[train_index], xdf.values[test_index]
        y_train, y_test = ydf.values[train_index], ydf.values[test_index]
        clf.fit(x_train, y_train)
        pre_val = clf.predict(x_test)
        accuracy = accuracy_score(y_test, pre_val)
        scores.append(accuracy)
        print("교차 검증 정확도 [{}] {} ".format(iter_count, accuracy))
    mean_score = np.mean(scores)
    print("평균 정확도: ", mean_score)
    
def transform_features(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Age_cate'] = df['Age'].apply(lambda x:get_category(x))
    df = encode_features(df, ['Cabin', 'Sex', 'Embarked', 'Age_cate'])
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Age'], axis=1)
    
    return df

class MyDummyClassifier(BaseEstimator):
    def fit(self, x, y = None):
        pass
    def predict(self, x):
        pred = np.zeros((x.shape[0], 1))
        
        for i in range(x.shape[0]):
            if x['Sex'].iloc[i] == 1:
                pred[i] = 0
            else:
                pred[i] = 1
        return pred
    
# 정확도
class MyFakeClassifier(BaseEstimator):
    def fit(self, x, y = None):
        pass
    def predict(self, x):
        return np.zeros((len(x), 1), dtype = bool)
    
# 정밀도와 재현율
def get_clf_eval(actual, pred):
    confusion = confusion_matrix(actual, pred)
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    
    return accuracy, precision, recall
def get_eval_by_thresholds(y_test, pred, thresholds):
    for threshold in thresholds:
        binar = Binarizer(threshold=threshold)
        pv = binar.fit_transform(pred)
        acc, prec, recall = get_clf_eval(y_test, pv)
        
        print(f'acc:{acc:.3f} prec:{prec:.3f} recall:{recall:.3f}')
        
# F1 Score
def get_clf_eval_f1(actual, pred):
    con = confusion_matrix(actual, pred)
    acc = accuracy_score(actual, pred)
    pre = precision_score(actual, pred)
    rec = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    
    print(f'오차 행렬\n {con}')
    print(f'정확도:{acc:6.4f} 정밀도:{pre:6.4f} 재현율:{rec:6.4f} F1:{f1:6.4f}')
def get_eval_by_thresholds_f1(y_test, pred, thresholds):
    for threshold in thresholds:
        binar = Binarizer(threshold=threshold)
        pv = binar.fit_transform(pred)
        get_clf_eval_f1(y_test, pv)
        
# ROC 곡선과 AUC
def roc_curve_plot(y_test, pred):
    fprs, tprs, thresholds = roc_curve(y_test,pred)
    plt.plot(fprs, tprs, label="ROC")
    plt.plot([0,1], [0,1], 'k--', label='Random')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()
def get_clf_eval_roc(actual, pred=None, pred_proba=None):
    con = confusion_matrix(actual, pred)
    acc = accuracy_score(actual, pred)
    pre = precision_score(actual, pred)
    rec = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred_proba)
    
    print('오차 행렬\n', con)
    print(f'정확도;{acc:6.4f} 정밀도:{pre:6.4f} 재현율:{rec:6.4f} F1:{f1:6.4f}')
    print(f'ROC AUC:{roc_auc:6.4f}')