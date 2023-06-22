import numpy as np

def mse(actual, pred):
    return np.sum((actual-pred)**2/len(actual))

def gradient(x,y,w,b):
    pred = w * x + b
    error = y - pred
    n = len(y)
    wg = -2 * sum(x*error)/n # 기울기
    bg = -2 * sum(error)/n
    
    return wg, bg, mse(y,pred)
    
def gradient_desent(x,y,lr=0.001):
    w = np.random.uniform(-1,1)
    b = np.random.uniform(-10,10)
    hist=[]
    for i in range(10000):
        wg,bg,r = gradient(x,y,w,b)
        if (i<5)|(i%100==0):
            hist.append([i,round(w,3),round(wg,3),round(b,3),round(bg,3),r])
        w = w - wg*lr
        b = b - bg*lr
    return w,b,hist