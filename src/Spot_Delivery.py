import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
import random
import pickle
import matplotlib.pyplot as plt

%matplotlib inline
df =pd.read_csv("Spot_Delivery.csv")
data=df.copy()
data=data.drop(columns=["Higher APR or payment than negotiated","Remarks",])
data=data.drop(columns=["Service"])
for i in range(0,len(data)):
    if data["Spot Delivery Scam"][i]!=1.0:
        data["Spot Delivery Scam"][i]=0.0
data = data.set_index("Sn")
indexname = data[ ~data['Spot Delivery Scam'].isnull() ].index
index_spot= data[data["Spot Delivery Scam"]==1.0].index
index_non_spot=data[data["Spot Delivery Scam"]==0.0].index
sampling = random.choices(index_non_spot, k=100)
index_spot
for i in index_spot : 
    sampling.append(i) 
len(sampling)
import random
z=random.shuffle(sampling)
indexname
tf=pd.read_csv("tfidfneg_max20.csv")
tf
tf=tf.set_index("Index")
sampling
tf_x=tf.loc[sampling]
index_test = data[ data['Spot Delivery Scam'].isnull() ].index

data_spot=data.loc[index_spot]

data_spot.to_csv("Spot Delivery Scam.csv")

tf_test=tf.loc[index_test]

data_l=data.loc[indexname]
data_sample=data.loc[sampling]
Y=data_sample["Spot Delivery Scam"][0:len(data_sample)]
z_new=pd.read_csv("tf_idf_synthetic.csv")

z_new=z_new.set_index("Sn")

label_synthetic=pd.read_csv("Y_synthetic.csv")

label_synthetic=label_synthetic.set_index("Sn")

label_synthetic
frames=[tf_x,z_new]
result=pd.concat(frames)
result

res_y.to_csv("temp_spot.csv")#Made changes- compiled two columns into one 
result_y=pd.read_csv("temp_spot.csv") #Use this
result_y=result_y.set_index("Sn")
X=result.to_numpy()
Y=result_y['0'].to_numpy()
result_y['0']
X.shape
Y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

clf = svm.SVC(kernel='linear', C = 1.0,probability = True)
clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)
y_pred
y_test
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
grid.fit(X,Y)

grid.best_params_
filename = 'Spot_Delivery.sav'
pickle.dump(grid, open(filename, 'wb'))
