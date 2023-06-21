from sklearn.preprocessing import StandardScaler
import numpy as np

# 사이킷런의 StandardScaler를 이용해 정규 분포 형태로 Amount Feature value를 변환하는 로직으로 수정
'''
def get_preprocessed_df(df=None):
    df_tmp = df.copy()
    scaler = StandardScaler()
    amount_n = scaler.fit_transform(df_tmp['Amount'].values.reshape(-1,1))
    
    # 변환된 Amount를 Amount_Scaled로 피쳐명 변경후 Dataframe 맨 앞 칼럼에 입력
    df_tmp.insert(0, 'Amount_Scaled', amount_n)
    
    # 기존 Time, Amount 피쳐 삭제
    df_tmp.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    return df_tmp
'''
def get_preprocessed_df(df=None):
    df_tmp = df.copy()
    
    # numpy의 log1p()를 이용해 Amount를 log 변환
    amount_n = np.log1p(df_tmp['Amount'])
    
    # 변환된 Amount를 Amount_Scaled로 피쳐명 변경후 Dataframe 맨 앞 칼럼에 입력
    df_tmp.insert(0, 'Amount_Scaled', amount_n)
    
    # 기존 Time, Amount 피쳐 삭제
    df_tmp.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    return df_tmp
def get_preprocessed_df2(df=None):
    df_tmp = df.copy()
    amount_n = np.log1p(df_tmp['Amount'])
    df_tmp.insert(0, 'Amount_Scaled', amount_n)
    df_tmp.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # 이상치 데이터 삭제
    outlier_index = get_outlier(df=df_tmp, column='V14', weight=1.5)
    df_tmp.drop(outlier_index, axis=0, inplace=True)
    
    return df_tmp
    

def get_outlier(df=None, column=None, weight=1.5):
    # fraud에 해당하는 column 데이터만 추출, 1/4 분위와 3/4 분위 지점을 np.percentile로 계산
    fraud = df[df['Class']==1][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
    
    # IQR을 구하고, IQR에 1.5를 곱ㅈ해 최댓값과 최솟값 지점을 구함
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 - iqr_weight
    
    # 최댓값보다 크거나, 최솟값보다 작은 값을 이상치 데이터로 설정, Dataframe index 전환
    outlier_index = fraud[(fraud<lowest_val)|(fraud>highest_val)].index
    
    return outlier_index