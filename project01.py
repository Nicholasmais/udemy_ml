import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import random

df = pd.read_csv("car_data.csv")

x = df.iloc[:,[1,3,4,6]].values
y = df.iloc[:,2].values
most_likes = df.iloc[:,-1].values

encoder_fuel = LabelEncoder()
x[:,2] = encoder_fuel.fit_transform(x[:,2])
encoder_km = LabelEncoder()
x[:,3] = encoder_km.fit_transform(x[:,3])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

regressor = RandomForestRegressor()
regressor.fit(x_train,y_train)
y_predicted = regressor.predict(x_test)


import matplotlib.pyplot as plt
x2 = [] 

for index in range(0,len(y_predicted)):
    x2.append(index)
    
plt.scatter(x2,y_test/1000, label='Esperado')
plt.scatter(x2,y_predicted/1000, label='Obtido', alpha=0.5)
plt.xlabel("Index")
plt.ylabel("Preço de venda (KR$)")


year = df['year'].unique()
km_driven = df['km_driven'].unique()
fuel = df['fuel'].unique()
transmission = df['transmission'].unique()

numbers = random.randint(500,1001)
x_tes2 = []


for key in range(0,numbers):
    random_value = [int(random.choice(year)),int(random.choice(km_driven)),str(random.choice(fuel)),str(random.choice(transmission))]
    x_tes2.append(random_value)

from numpy import array
x_tes2 = array(x_tes2)
for row in range(len(x_tes2)):
    print(row, x_tes2[row])
encoder_fuel2 = LabelEncoder()
x_tes2[:,2] = encoder_fuel2.fit_transform(x_tes2[:,2])
encoder_km2 = LabelEncoder()
x_tes2[:,3] = encoder_km2.fit_transform(x_tes2[:,3])


y_predicted2 = regressor.predict(x_tes2)

plt.scatter([number for number in range(len(y_predicted2))],y_predicted2/1000, label='Expeculado', color='r', alpha=0.5)


plt.legend()
plt.show()

