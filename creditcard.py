from sklearn.model_selection import train_test_split
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

# 인자로 입력받은 DataFrame을 복사한 뒤 Time 칼럼만 삭제하고 복사된 DataFrame 반환
def get_preprocessed_df(df=None):
    df_tmp = df.copy()
    df_tmp.drop('Time', axis=1, inplace=True)
    
    return df_tmp

# 사전 데이터 가공 후 학습과 테스트 데이터 세트를 반환하는 함수
def get_train_test_dataset(df=None):
    # 인자로 입력된 DataFrame의 사전 데이터 가공이 완료된 복사 DataFrame 반환
    df_tmp = get_preprocessed_df(df)
    
    # DataFrame의 맨 마지막 칼럼이 레이블, 나머지는 피처들
    x_features = df_tmp.iloc[:,:-1]
    y_target = df_tmp.iloc[:,-1]
    
    # train_test_split으로 학습과 테스트 데이터 분할
    ## stratify=y_target으로 Stratified 기반 분할
    x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size=0.3, random_state=0, stratify=y_target)
    
    return x_train, x_test, y_train, y_test

# 인자로 사이킷런의 Estimator 객체와 학습/테스트 데이터 세트를 입력받아서 학습/예측/평가 수행
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:,1]
    get_clf_eval(tgt_test, pred, pred_proba)