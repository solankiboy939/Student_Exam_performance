import pandas as pd
import numpy as np
data = pd.read_csv("project_on_student_prediction\student_exam_data.csv")
data.shape
data.head()
data.dtypes
data.columns
x = data.drop(['Pass/Fail'], axis=1)
y = data['Pass/Fail']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
from sklearn.metrics import accuracy_score
model.fit(x_train_scaler, y_train)
y_pred = model.predict(x_test_scaler)
print("Accuracy : ",accuracy_score(y_test, y_pred))