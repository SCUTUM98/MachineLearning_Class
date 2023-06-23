import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# log 값 변환시 NaN 이슈 발생으로 log()가 아닌 log1p()를 이용
# RMSLE 계산
def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    se = (log_y - log_pred)**2
    
    return np.sqrt(np.mean(se))

# RMSE 계산
def rmse(y, pred):
    mse = np.sqrt(mean_squared_error(y, pred))
    
    return mse

# MSE, RMSE, RMSLE 계산
def evaluate_regr(y, pred):
    rmsle_val = rmsle(y, pred)
    rmse_val = rmse(y, pred)
    mae_val = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    
    print(f'r2: {r2:.3f} RMSLE: {rmse_val:3f} RMSE:{rmse_val:.3f} MAE: {mae_val:.3f}')
    
def get_top_error_data(y_test, pred, n_tops=5):
    # DataFrame의 칼럼으로 실제 대여 횟수와 예측값을 서로 비교할 수 있도록 생성
    rdf = pd.DataFrame(y_test.values, columns=['real_count'])
    rdf['predicted_count'] = np.round(pred)
    rdf['diff'] = np.abs(rdf['real_count']-rdf['predicted_count'])
    
    # 예측 값과 실제 값이 가장 큰 데이터 순으로 출력
    print(rdf.sort_values('diff', ascending=False)[:n_tops])
    
# 모델과 학습/테스트 데이터 세트를 입력하여 성능 평가 수치 반환
def get_model_predict(model, x_train, x_test, y_train, y_test, is_expm1=False):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    
    if is_expm1:
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)
    print('###', model.__class__.__name__, "###")
    evaluate_regr(y_test, pred)