# Imported All Important Libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
# Train Dataset
df=pd.read_csv("training_titanic_x_y_train.csv")
# Removed Name , Ticket , Cabin Columns from the Trained Dataset and Cleaned it
df_dropped = df.drop(columns=['Name','Ticket','Cabin'])
# Changed  Male to 1 and Women to 0 for better computation
df_dropped['Sex'] = df_dropped['Sex'].apply(lambda x: 1 if x == 'male' else 0)
# Changed C to 1 , Q to 2 and S to 3
df_dropped['Embarked'] = df_dropped['Embarked'].apply(lambda x: 1 if x == 'C' else (2 if x == 'Q' else 3))
# Filled Nan values of Age with Mean values
df_dropped['Age'].fillna(df_dropped['Age'].mean(),inplace=True)
# Test Dataset
ds=pd.read_csv("All new Titanic Test data set.csv")
# Removed Name , Ticket , Cabin Columns from the Testing Dataset and Cleaned it
ds_dropped = ds.drop(columns=['Name','Ticket','Cabin'])
# Changed  Male to 1 and Women to 0 for better computation
ds_dropped['Sex'] = ds_dropped['Sex'].apply(lambda x: 1 if x == 'male' else 0)
# Changed C to 1 , Q to 2 and S to 3
ds_dropped['Embarked'] = ds_dropped['Embarked'].apply(lambda x: 1 if x == 'C' else (2 if x == 'Q' else 3))
# Filled all Nan values of the Testdataset
ds_dropped['Age'].fillna(ds_dropped['Age'].mean(),inplace=True)
X_train = df_dropped.iloc[:, :7]
Y_train = df_dropped.iloc[:,-1]
# Changed The value in array format
x_train=X_train.values
y_target=Y_train.values
x_test=ds_dropped.values
# used The LogisticRegression To fit and Predict The Trained and Test DataSets
clf = LogisticRegression()
clf.fit(x_train,y_target)
# Predicted The Test Data
y_pred=clf.predict(x_test)
# Calculated The Score
clf.score(x_train,y_target)
