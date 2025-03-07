import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt

%matplotlib inline
df =pd.read_csv("Sales_neg.csv")
data=df.copy()
data=data.drop(columns=["Higher APR or payment than negotiated","Remarks",])
data=data.drop(columns=["Service"])
data = data.set_index("Sn")
indexname = data[ data['Labelled'] ==1  ].index
indexadvert = data[ data['Advertised Price discrepancy'] ==1  ].index
z=pd.read_csv("tfidfneg_max20.csv")
z=z.set_index("Index")
tf=z
#tf=tf.set_index("Index")

tf_x=tf.loc[indexname]

index_test = data[ data['Labelled'] !=1  ].index

tf_test=tf.loc[index_test]

data_l=data.loc[indexname]
data_l
data_shuf=data_l.sample(frac=1)
index_shuf=data_shuf.index
tf_x_shuf=z.loc[index_shuf]
X=tf_x_shuf.to_numpy()
Y=data_shuf["Advertised Price discrepancy"]
y=Y.to_numpy()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

clf = svm.SVC(kernel='linear', C = 1.0,probability = True)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
confusion_matrix(y_test, y_pred)
f1_score(y_test, y_pred, average='micro')
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}

grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)

grid.fit(X_train,y_train)
grid.best_params_
predic = grid.predict(X_test)

print(classification_report(y_test,predic))
print(confusion_matrix(y_test, predic))
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    grid.fit(X_train,y_train)
    predic = grid.predict(X_test)

    print(classification_report(y_test,predic))
    print(confusion_matrix(y_test, predic))
grid.fit(X,y)
x_texas=pd.read_csv("Texas_Neg_Tf-idf_without_threshold.csv")
x_texas
x_texas=x_texas.set_index("Unnamed: 0")

X_test_texas=x_texas.to_numpy()
X_test_texas.shape
predic = grid.predict(X_test_texas)
res_tex=pd.DataFrame({"Label":predic})
res_tex=res_tex.set_index(x_texas.index)
res_tex
res_ind=res_tex[res_tex["Label"]==1.0].index
res_ind
import glob

path = r'Texas_without_threshold' # use your path
all_files = glob.glob(path + "/*.csv")

li = []
names=[]
l=0
for filename in all_files:
    l+=1
    df = pd.read_csv(filename, index_col=None, header=0)
    df['Serial']=l
    #names.append(filename)
    df["Name"]=filename
    
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame
l=frame.loc[res_ind]
l
l.to_csv("Advertised_Price_Results_Texas.csv")
filename = 'Advertised_Prices.sav'
pickle.dump(grid, open(filename, 'wb'))
