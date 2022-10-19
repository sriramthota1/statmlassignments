import pandas as pd
d=pd.read_csv(r"C:\Users\Sriram Thota\Desktop\Salary_Data-1.csv")
print(d)
print("\n")
x=d.iloc[:,0:1]
y=d.iloc[:,1:2]
print(x,"\n",y)
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=5)
print(X_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import Lasso
model_lasso= Lasso(alpha=0.01)
model_lasso.fit(x,y)
pred_train_lasso= model_lasso.predict(x_test)
print(mean_squared_error(y_test,pred_train_lasso))
from sklearn.linear_model import Ridge
rr= Ridge(alpha=0.01)
rr.fit(X_train, y_train)
pred_train_rr= rr.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train,pred_train_rr))