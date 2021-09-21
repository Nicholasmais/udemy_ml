import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
df = pd.read_csv("candy-data.csv")


df['Most people like it?'] = np.where(df['winpercent']>=55,1,0)
df['Price good?'] = np.where(df['pricepercent']>=.5,0,1)
df['Sugar good?'] = np.where((df['sugarpercent']>=.25) & (df['sugarpercent']<=0.75),1,0)

x = df.iloc[:,[1,2,3,4,5,6,7,8,9,14,15]].values
y = df.iloc[:,-3].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)

regressor = LogisticRegression()

regressor.fit(x_train,y_train)

y_predicted = regressor.predict(x_test)

numbers = [num for num in range(len(y_predicted))]
print(numbers)
def sigmoid(x, x0, k):
    y = 1 / (1+ np.exp(-k*(x-x0)))
    return y

popt, pcov = curve_fit(sigmoid, numbers, y_predicted)

x = np.linspace(0, 9, 10)
y = sigmoid(x, *popt)

plt.plot(numbers, np.sort(y_predicted), 'o', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.ylim(0, 1.25)
plt.legend(loc='best')
 
plt.show()

