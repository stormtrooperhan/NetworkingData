import pandas as pd
import numpy as np

df = pd.read_csv("datasets_for_paper.csv", low_memory=True)


##firstPaint provides time info about page rendering 
print(df.dtypes)
df["nodeId"]=df["nodeId"].astype(float)
df["numObj"]=df["numObj"].astype(float)
df["firstPaint"]=df['firstPaint'].astype(float)

##def changeProtName(value):
##    if value== 'H1':
##        return H3
##    else:
##        return value
##
##df['protocol'] = df['protocol'].map(lambda x: changeProtName(x))


df["protocol"] = df["protocol"].astype('category')
print(df['protocol'])

#dont change protocol to float 

#df=pd.get_dummies(df, columns=['browser', 'nodeType', 'protocol','url'])

df=pd.get_dummies(df)

print(df.dtypes)

##labels=np.array(df['protocol'])
##df=df.drop('protocol', axis=1)
##df_list=list(df.columns)
##df=np.array(df)
##
##from sklearn.model_selection import train_test_split
##train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size = 0.25, random_state = 42)
##print('Training Features Shape:', train_features.shape)
##print('Training Labels Shape:', train_labels.shape)
##print('Testing Features Shape:', test_features.shape)
##print('Testing Labels Shape:', test_labels.shape)



