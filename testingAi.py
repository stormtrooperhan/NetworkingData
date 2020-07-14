import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics 
from sklearn import model_selection

from sklearn import linear_model



from sklearn.model_selection import train_test_split
from sklearn import preprocessing as preprocessing

from sklearn.metrics import accuracy_score
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import KFold


from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#kfold_5 = KFold(n_splits = numFolds, shuffle=True)

mpl.rcParams['figure.dpi']=400

df = pd.read_csv("datasets_for_paper.csv", low_memory=False)

##firstPaint provides time info about page renderingso does ,rumSpeedIndex=avg page render
print(df.dtypes)
df["nodeId"]=df["nodeId"].astype(int)
df["numObj"]=df["numObj"].astype(int)
df["rumSpeedIndex"]=df['rumSpeedIndex'].astype(int)
df["pageLoadTime"]=df['pageLoadTime'].astype(int)
df["firstPaint"]=df['firstPaint'].astype(int)

#convert from name into pure string
def changeProtName(value):
    if value== 'H1s':
        return str('Hs')
    else:
        return str('Hl')

df['protocol'] = df['protocol'].map(lambda x: changeProtName(x))

#hot encode catagories as catagorical data 
df['protocol']=pd.Categorical(df["protocol"])
df['browser']=pd.Categorical(df['browser'])
df['nodeType']=pd.Categorical(df['nodeType'])
df['url']=pd.Categorical(df['url'])

#list a bunch of details about categorical data 
def summerize_data(df1):
    for column in df1.columns:
        print(column)
        if df.dtypes[column] == np.object: 
            print(df1[column].value_counts())
        else:
            print(df1[column].describe())
            
        print('\n')
    
summerize_data(df)

def hotEncodingCats(df1):
    results=df1.copy()
    encoders={}
    for column in results.columns:
        encoders[column]=preprocessing.LabelEncoder()
        results[column]=encoders[column].fit_transform(results[column])
    return results, encoders


print(df.dtypes)

encoded_data, _ = hotEncodingCats(df)
sns.heatmap(encoded_data.corr(), square=True)
#plt.show()

encoded_data.tail(5)

encoded_data, encoders = hotEncodingCats(df)
new_series = encoded_data["protocol"]

X_train, X_test, y_train, y_test = train_test_split(encoded_data[encoded_data.columns.drop("protocol")], new_series, train_size=0.70)
scaler = preprocessing.StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = scaler.transform(X_test)

cls = linear_model.LogisticRegression()

cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)

print(df.types)
print(accuracy_score(y_test,y_pred))
print(df.dtypes)








    


#labels=np.array(df['protocol'])
#df=df.drop('protocol', axis=1)
#df_list=list(df.columns)
#df=np.array(df)
##
##from sklearn.model_selection import train_test_split
##train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size = 0.25, random_state = 42)
##print('Training Features Shape:', train_features.shape)
##print('Training Labels Shape:', train_labels.shape)
##print('Testing Features Shape:', test_features.shape)
##print('Testing Labels Shape:', test_labels.shape)



