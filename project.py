import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score
data = pd.read_csv("iris.csv")

x = data.iloc[:,1:5].values
y = data.iloc[:,5].values

binary = LabelBinarizer()

y_bin = binary.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y_bin, test_size=0.2, random_state=42)

classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
ypred = classifier.predict(x_test)

precision = precision_score(y_true=y_test,y_pred=ypred, average="micro")
recall = recall_score(y_true=y_test,y_pred=ypred,average='micro')
accuracy = accuracy_score(y_true=y_test,y_pred=ypred)

print(f"Precision Score = {precision}")
print(f"Recall score = {recall}")
print(f"Accurcy score = {accuracy}")
