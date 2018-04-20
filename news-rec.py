#Jonas Albaira - Assignment 3
from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
palette = sns.color_palette('deep', 5)
palette[1], palette[2] = palette[2], palette[1]

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestRegressor

news_label = pd.read_csv('ctrain.csv', header=None) #csv with labels

#news_train_data = pd.read_csv('training.csv', header=none) #train data, no headers
news_train_data = pd.read_csv('training2.csv', header=None) #train data, no headers, title content

#newsLabel = news_train_data[0]

#X_train = news_train_data.drop([0], axis=1)
X_train = news_train_data
y_train = news_label[0]

RFR = LogisticRegression()
RFR.fit(X_train, y_train)

predictions = RFR.predict(X_train)

newsLabel = pd.read_csv('test2.csv', header=None) #csv with labels
#X_test = newsLabel

predictionsTitle = RFR.predict(newsLabel)

count = []
for i in range(1, len(predictionsTitle)+1):
    count.append(i)

competition_entry = pd.DataFrame({"id":count, "class": predictionsTitle})
competition_entry.to_csv("competition-entry.csv", index=False)






