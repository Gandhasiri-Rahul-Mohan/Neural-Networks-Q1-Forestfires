# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 14:25:41 2023

@author: Rahul
"""

#importing required librarires
# pip install imblearn
# pip install tensorflow


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense,Dropout
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam,RMSprop
from sklearn.model_selection import GridSearchCV,KFold
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Neural Networks\\forestfires.csv")
df

df.shape
df.isna().sum()
df.columns
df.describe()

for i in df.columns:
    print(df[i].value_counts())
    print()

#Visualization of Data parameters
sns.countplot(df['size_category'])
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = df['month'],hue=df['size_category'])
plt.show()

plt.figure(figsize=(10,8))
sns.countplot(x = df['day'],hue=df['size_category'])
plt.show()

plt.rcParams["figure.figsize"] = 9,5
plt.figure(figsize = (12,6))
print("Skew: {}".format(df['area'].skew()))
print("Kurtosis: {}".format(df['area'].kurtosis()))
ax = sns.kdeplot(df['area'],shade = True,color = 'orange')
plt.xticks([i for i in range(0,1200,50)])
plt.show()

#correlation 
corr = df[df.columns[0:11]].corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr,annot = True)

dfa = df[df.columns[0:10]]
month_colum = dfa.select_dtypes(include = 'object').columns.tolist()
num_columns = dfa.select_dtypes(exclude = 'object').columns.tolist()
plt.figure(figsize = (20,15))

for i,col in enumerate(num_columns,1):
    plt.subplot(8,4,i)
    sns.kdeplot(df[col],color = 'orange',shade = True)
    plt.subplot(8,4,i+10)
    df[col].plot.box()
plt.tight_layout() 
plt.show()
num_data = df[num_columns]
pd.DataFrame(data = [num_data.skew(),num_data.kurtosis()],index = ['skewness','kurtosis'])

# natural logarithm scaling (+1 to prevent errors at 0)
df.loc[:, ['rain', 'area']] = df.loc[:, ['rain', 'area']].apply(lambda x: np.log(x + 1), axis = 1)
fig, ax = plt.subplots(2, figsize = (5, 8))
ax[0].hist(df['rain'])
ax[0].title.set_text('histogram of rain')
ax[1].hist(df['area'])
ax[1].title.set_text('histogram of area')

drop_data = df.drop(labels=['month','day'],axis = 1)
drop_data

le = LabelEncoder()
drop_data['size_category'] = le.fit_transform(drop_data['size_category'])
drop_data

sns.countplot(drop_data['size_category'])

mapping = {'small': 1, 'large': 2}
drop_data = drop_data.replace(mapping)

#split the data into x and y

x = drop_data.drop(labels='size_category',axis = 1)
y = drop_data[['size_category']]

x_train, x_test, y_train, y_test  = train_test_split(x,y,test_size=0.30,random_state=12)

#Data Is Imbalance so i have to balance it , so i m using here smote operation of balancing technique
sm = SMOTE(random_state=12)
x_train_sm,y_train_sm = sm.fit_resample(x_train,np.array(y_train).ravel())
x_train_sm,y_train_sm

x_train = x_train_sm.copy()
y_train = y_train_sm.copy()

#Convert Data into standard scale
scale = MinMaxScaler()
X_train = scale.fit_transform(x_train)
X_train

X_test = scale.fit_transform(x_test)
X_test

#Model Training
#Tuning of Hyperparameter : Batch size and Epoch

def creat_model():
    model = Sequential()
    model.add(Dense(8, input_dim = 28,kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4,kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,kernel_initializer='uniform', activation='sigmoid'))
    adam = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',optimizer = adam,metrics='accuracy')
    return model

model = KerasClassifier(build_fn=creat_model,verbose = 0)
batch_size = [10,30,50]
epochs = [10,20,50]
param_grid = dict(batch_size = batch_size,epochs = epochs)
gsv = GridSearchCV(estimator=model,param_grid=param_grid,cv = KFold(),verbose=5)
gsv_res = gsv.fit(X_train,y_train)

print(gsv_res.best_params_,gsv_res.best_score_)

#Turning Hyperparameter: Learning rate and Dropout rate
def creat_model(learning_rate,dropout_rate):
    model = Sequential()
    model.add(Dense(8, input_dim = 28,kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4,kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,kernel_initializer='uniform', activation='sigmoid'))
    adam = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy',optimizer = adam,metrics='accuracy')
    return model
model = KerasClassifier(build_fn=creat_model,batch_size = 10,epochs = 50,verbose = 0)
learning_rate = [0.1,0.01,0.001]
dropout_rate = [0.0,0.1,0.2]
param_grid = dict(learning_rate = learning_rate,dropout_rate = dropout_rate)
gsv = GridSearchCV(estimator=model,param_grid=param_grid,cv= KFold(),verbose=5)
gsv_r = gsv.fit(X_train,y_train)

print(gsv_r.best_params_,gsv_r.best_score_)

#Tuning of Hyperparameter :Activation Function & Kernel Initializer
def creat_model(Activation_Function,init):
    model = Sequential()
    model.add(Dense(8, input_dim = 28,kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(4,kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1,kernel_initializer='uniform', activation='sigmoid'))
    adam = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',optimizer = adam,metrics='accuracy')
    return model
model = KerasClassifier(build_fn=creat_model,batch_size = 10,epochs = 50,verbose = 0)
Activation_Function = ['relu','tanh','softmax','linear']
init = ['zero','uniform','normal']
param_grid = dict(Activation_Function = Activation_Function,init = init)
gsv = GridSearchCV(estimator=model,param_grid=param_grid,cv= KFold(),verbose=5)
gsv_result = gsv.fit(X_train,y_train)

print(gsv_result.best_score_,gsv_result.best_params_)

#Tuning of Hyperparameter :Number of Neurons in hidden layer
def creat_model(neuron1,neuron2):
    model = Sequential()
    model.add(Dense(8,input_dim=28,kernel_initializer='normal', activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(4,kernel_initializer='normal',activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    adam = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn=creat_model,batch_size = 10,epochs = 50,verbose = 0)
neuron1 = [24,16,8]
neuron2 = [12,8,4]
param_grid = dict(neuron1 = neuron1,neuron2=neuron2)
gsv = GridSearchCV(estimator=model,param_grid=param_grid,cv=KFold(),verbose=5)
gsv_n = gsv.fit(X_train,y_train)

print(gsv_n.best_score_,gsv_n.best_params_)

#Train a model with optimum values of hyperparameter
# best Parameters
def creat_model():
    model = Sequential()
    model.add(Dense(24,input_dim=28,kernel_initializer='normal', activation='softmax'))
    model.add(Dropout(0.1))
    model.add(Dense(8,kernel_initializer='normal',activation='softmax'))
    model.add(Dropout(0.1))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    adam = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn=creat_model,batch_size = 10,epochs = 50)
model.fit(X_train,y_train)

# testing data
y_test_pred = model.predict(X_test)
accuracy_score(y_test,y_test_pred)

confusion_matrix(y_test,y_test_pred)

print(classification_report(y_test,y_test_pred))






























































































