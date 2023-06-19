# 지니 계수 계산 방법
def gini2(datas):
    total = 0
    for data in datas:
        total += data
        
    s = 0
    for data in datas:
        s += (data/total)**2
        
    return 1-s