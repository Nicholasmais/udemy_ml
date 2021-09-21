import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv("creditcard.csv")

x = df.iloc[:,1:29].values
y = df.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

gnb = GaussianNB()

gnb.fit(x_train,y_train)
y_predicted = gnb.predict(x_test)
  
print(accuracy_score(y_test,y_predicted))
fig, (ax1,ax2) = plt.subplots(2,1)
ax1.hist(df.loc[df["Class"]==0,"Amount"].values[:],color='g', stacked=False, bins=np.arange(0,len(df.loc[df["Class"]==0,"Amount"].values[:]),1))
ax2.hist(df.loc[df["Class"]==1,"Amount"].values[:],color='r', stacked=False, bins=np.arange(0,len(df.loc[df["Class"]==0,"Amount"].values[:]),1))
plt.show()