from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def get_category(age):
    cate = ""
    if age<0:
        cate = 'Unknown'
    elif age <= 5:
        cate = "B"
    elif age <= 12:
        cate = "C"
    elif age <= 18:
        cate = "T"
    elif age <= 25:
        cate = "S"
    elif age <= 35:
        cate = "Y"
    elif age <= 60:
        cate = "A"
    else:
        cate = "E"
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

def exec_kfold(clf, xdf, ydf, folds = 5):
    kfold = KFold(n_splits=folds)
    
    for iter_count, (train_index, test_index) in enumerate(kfold.split(xdf)):
        x_train, x_test = xdf.values[train_index], xdf.values[test_index]
        y_train, y_test = ydf.values[train_index], ydf.values[test_index]
        clf.fit(x_train, y_train)
        pre_val = clf.predict(x_test)
        print(accuracy_score(y_test, pre_val))