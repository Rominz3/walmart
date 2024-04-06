#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from category_encoders import BinaryEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# # EDA

# In[2]:


data=pd.read_csv(R'F:\job\New folder\Walmart.csv')


# In[3]:


data.isnull().sum()


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


def seasonName(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'
    
data['Date'] = pd.to_datetime(data['Date'], format = "%d-%m-%Y")
data['Month_Name'] = data['Date'].dt.month_name()
data['Season'] = data['Date'].dt.month.apply(seasonName)
data['Week'] = data['Date'].dt.isocalendar().week.astype('int32')


# In[8]:


data.head(20)


# In[9]:


numericalData = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
plt.figure(figsize=(10,10))

for index,col in enumerate(numericalData):
    plt.subplot(3,2,index+1)
    sns.histplot(data=data,x=col,kde=True,bins=15)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
    


# In[10]:


Holiday_Flag_counts=data['Holiday_Flag'].value_counts()
print(Holiday_Flag_counts)


# In[11]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(data=data,x='Holiday_Flag')
plt.xticks([0,1], ["NonHolidays", "Holidays"])
plt.title('Total Number of Holidays and Non-Holidays')
plt.subplot(1,2,2)
plt.pie(x=Holiday_Flag_counts,autopct='%1.1f%%',labels=Holiday_Flag_counts.index)
plt.title("Percentage Distribution of Holidays")
plt.show()


# In[12]:


totalSalesBySeason=data.groupby('Season')['Weekly_Sales'].sum()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.barplot(data=data,x=totalSalesBySeason.index,y=totalSalesBySeason)
for i, (season, value) in enumerate(totalSalesBySeason.items()):
    plt.annotate('${:.2f}'.format(value), (i, value), ha='center', va='bottom', fontsize=8)
plt.title('Total Sales by Season')


plt.subplot(1,2,2)
plt.pie(data=data, x=totalSalesBySeason,labels=totalSalesBySeason.index,autopct='%1.1f%%')
plt.title('Distribution of Total Sales by Season')
plt.tight_layout()
plt.show


# In[13]:


totalSalesByHolidayFlag=data.groupby('Holiday_Flag')['Weekly_Sales'].sum()
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.barplot(data=data,x=totalSalesByHolidayFlag.index,y=totalSalesByHolidayFlag)
for i, (HolidayFlag, value2) in enumerate(totalSalesByHolidayFlag.items()):
    plt.annotate('${:.2f}'.format(value2), (i, value2), ha='center', va='bottom', fontsize=8)
plt.title('Total Sales by Holiday Flag')   
plt.subplot(1,2,2)
plt.pie(data=data, x=totalSalesByHolidayFlag,labels=totalSalesByHolidayFlag.index,autopct='%1.1f%%')
plt.title('Distribution of Total Sales by Holiday Flag')
plt.tight_layout()
plt.show


# In[14]:


totalSalesByStore = data.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 5))
sns.barplot(data=data, x=totalSalesByStore.index, y=totalSalesByStore,order=totalSalesByStore.index)
plt.title('Total Sales by Store')
plt.xlabel('Store Number')
plt.ylabel('Total Sales')
plt.show()
highestSalesStore=totalSalesByStore.idxmax()
highestSalesValue=totalSalesByStore.max()

lowestSalesStore=totalSalesByStore.idxmin()
lowesttSalesValue=totalSalesByStore.min()
print(f'highest sales store:{highestSalesStore}, total sales:${highestSalesValue}')
print(f'lowest sales store:{lowestSalesStore}, total sales:${lowesttSalesValue}')


# In[15]:


correlationMap=data.corr()
plt.figure(figsize=(8,5))
sns.heatmap(correlationMap,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# # Data Preprocessing

# In[16]:


data.isnull().sum()


# In[17]:


data.dtypes


# In[18]:


dataPreprocess = data.copy()
dataPreprocess.drop(['Date'], axis = 1, inplace = True)
dataPreprocess.dtypes


# In[19]:


dataPreprocess['Store']=dataPreprocess['Store'].astype('object')
dataPreprocess['Week']=dataPreprocess['Week'].astype('object')
dataPreprocess['Holiday_Flag']=dataPreprocess['Holiday_Flag'].astype('object')
dataPreprocess.dtypes


# In[20]:


X = dataPreprocess.drop('Weekly_Sales', axis = 1)
y = dataPreprocess['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

numericalFeatures = dataPreprocess.select_dtypes('number').columns.to_list()
numericalFeatures.remove('Weekly_Sales')

categoricalFeatures = dataPreprocess.select_dtypes('object').columns.to_list()

print(f"Numerical Features  : {numericalFeatures}")
print(f"Categorical Features: {categoricalFeatures}")


# In[21]:


preprocessor = ColumnTransformer([('num_features', StandardScaler(), numericalFeatures),
                                  ('cat_features', BinaryEncoder(), categoricalFeatures),])

preprocessor.fit(X_train)

X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


# # Model Building and Evaluation
# 

# # 1.Decision Tree Regressor

# In[22]:


decisionTree_regressor = DecisionTreeRegressor()
decisionTree_regressor.fit(X_train_transformed,y_train)
y_predict=decisionTree_regressor.predict(X_train_transformed)
MAE=mean_absolute_error(y_train,y_predict)
MSE=mean_squared_error(y_train,y_predict)
RMSE=np.sqrt(MSE)
R2=r2_score(y_train,y_predict)
print(f"Mean Absolute Error    : {MAE}")
print(f"Mean Squared Error     : {MSE}")
print(f"Root Mean Squared Error: {RMSE}")
print(f"\nR2 Score: {R2}")


# In[23]:


plt.figure(figsize=(8,4))
sns.kdeplot(x=y_train,color='blue',label='Actual Values')
sns.kdeplot(x=y_predict,color='orange',label='Predicted Values')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Distribution of Actual vs. Predicted Values')
plt.legend()
plt.show()


# # 2.Random Forest Regressor

# In[24]:


randomForest_regressor = RandomForestRegressor()
randomForest_regressor.fit(X_train_transformed,y_train)
y_predict=randomForest_regressor.predict(X_train_transformed)
MAE=mean_absolute_error(y_train,y_predict)
MSE=mean_squared_error(y_train,y_predict)
RMSE=np.sqrt(MSE)
R2=r2_score(y_train,y_predict)
print(f"Mean Absolute Error    : {MAE}")
print(f"Mean Squared Error     : {MSE}")
print(f"Root Mean Squared Error: {RMSE}")
print(f"\nR2 Score: {R2}")
plt.figure(figsize=(8,4))
sns.kdeplot(x=y_train,color='blue',label='Actual Values')
sns.kdeplot(x=y_predict,color='orange',label='Predicted Values')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Distribution of Actual vs. Predicted Values')
plt.legend()
plt.show()


# # 3.XGB Regressor

# In[25]:


XGB_regressor = XGBRegressor()
XGB_regressor.fit(X_train_transformed,y_train)
y_predict=XGB_regressor.predict(X_train_transformed)
MAE=mean_absolute_error(y_train,y_predict)
MSE=mean_squared_error(y_train,y_predict)
RMSE=np.sqrt(MSE)
R2=r2_score(y_train,y_predict)
print(f"Mean Absolute Error    : {MAE}")
print(f"Mean Squared Error     : {MSE}")
print(f"Root Mean Squared Error: {RMSE}")
print(f"\nR2 Score: {R2}")
plt.figure(figsize=(8,4))
sns.kdeplot(x=y_train,color='blue',label='Actual Values')
sns.kdeplot(x=y_predict,color='orange',label='Predicted Values')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Distribution of Actual vs. Predicted Values')
plt.legend()
plt.show()


# # 1. Decision Tree Regressor

# In[26]:


dt_parameter={'max_depth':np.arange(2,15),'min_samples_split':[10,20,30,40,50,100,200,300]}
grid_search=GridSearchCV(estimator=decisionTree_regressor,param_grid=dt_parameter,cv=5,scoring='r2')
grid_search.fit(X_train_transformed,y_train)
bestParameters=grid_search.best_params_
bestScore=grid_search.best_score_
decisionTree_regressor_tuned=grid_search.best_estimator_
print(f"Best parameters: {bestParameters} \n")
print(f"Best R2 score  : {bestScore}")


# In[27]:


decisionTree_regressor_tuned.fit(X_train_transformed,y_train)
y_predict=decisionTree_regressor_tuned.predict(X_train_transformed)
MAE=mean_absolute_error(y_train,y_predict)
MSE=mean_squared_error(y_train,y_predict)
RMSE=np.sqrt(MSE)
R2=r2_score(y_train,y_predict)
print(f"Mean Absolute Error    : {MAE}")
print(f"Mean Squared Error     : {MSE}")
print(f"Root Mean Squared Error: {RMSE}")
print(f"\nR2 Score: {R2}")


# In[28]:


plt.figure(figsize=(8,4))
sns.kdeplot(x=y_train,color='blue',label='Actual Values')
sns.kdeplot(x=y_predict,color='orange',label='Predicted Values')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Distribution of Actual vs. Predicted Values')
plt.legend()
plt.show()


# In[29]:


crossValScore=cross_val_score(estimator=decisionTree_regressor_tuned, X=X_train_transformed, y=y_train,scoring='r2',cv=5)
print(f"\nCross Validation Scores r2: {crossValScore}")
print(f"\nMean              : {crossValScore.mean() * 100:.2f}%")
print(f"Standard Deviation: {crossValScore.std()}")


# # 2. Random Forest Regressor

# In[30]:


rf_parameter={'max_depth':np.arange(2,15),'n_estimators':np.arange(25,101,25)}
grid_search=GridSearchCV(estimator=randomForest_regressor,param_grid=rf_parameter,cv=5,scoring='r2')
grid_search.fit(X_train_transformed,y_train)
bestParameters=grid_search.best_params_
bestScore=grid_search.best_score_
randomForest_regressor_tuned=grid_search.best_estimator_
print(f"Best parameters: {bestParameters} \n")
print(f"Best R2 score  : {bestScore}")


# In[31]:


randomForest_regressor_tuned.fit(X_train_transformed,y_train)
y_predict=randomForest_regressor_tuned.predict(X_train_transformed)
MAE=mean_absolute_error(y_train,y_predict)
MSE=mean_squared_error(y_train,y_predict)
RMSE=np.sqrt(MSE)
R2=r2_score(y_train,y_predict)
print(f"Mean Absolute Error    : {MAE}")
print(f"Mean Squared Error     : {MSE}")
print(f"Root Mean Squared Error: {RMSE}")
print(f"\nR2 Score: {R2}")


# In[32]:


plt.figure(figsize=(8,4))
sns.kdeplot(x=y_train,color='blue',label='Actual Values')
sns.kdeplot(x=y_predict,color='orange',label='Predicted Values')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Distribution of Actual vs. Predicted Values')
plt.legend()
plt.show()


# In[33]:


crossValScore=cross_val_score(estimator=randomForest_regressor_tuned, X=X_train_transformed, y=y_train,scoring='r2',cv=5)
print(f"\nCross Validation Scores r2: {crossValScore}")
print(f"\nMean              : {crossValScore.mean() * 100:.2f}%")
print(f"Standard Deviation: {crossValScore.std()}")


# # 3.XGB regressor

# In[34]:


xgb_parameter={'max_depth':np.arange(2,15),'n_estimators':np.arange(25,101,25)}
grid_search=GridSearchCV(estimator=XGB_regressor,param_grid=xgb_parameter,cv=5,scoring='r2')
grid_search.fit(X_train_transformed,y_train)
bestParameters=grid_search.best_params_
bestScore=grid_search.best_score_
xgb_regressor_tuned=grid_search.best_estimator_
print(f"Best parameters: {bestParameters} \n")
print(f"Best R2 score  : {bestScore}")


# In[35]:


xgb_regressor_tuned.fit(X_train_transformed,y_train)
y_predict=xgb_regressor_tuned.predict(X_train_transformed)
MAE=mean_absolute_error(y_train,y_predict)
MSE=mean_squared_error(y_train,y_predict)
RMSE=np.sqrt(MSE)
R2=r2_score(y_train,y_predict)
print(f"Mean Absolute Error    : {MAE}")
print(f"Mean Squared Error     : {MSE}")
print(f"Root Mean Squared Error: {RMSE}")
print(f"\nR2 Score: {R2}")


# In[36]:


plt.figure(figsize=(8,4))
sns.kdeplot(x=y_train,color='blue',label='Actual Values')
sns.kdeplot(x=y_predict,color='orange',label='Predicted Values')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Distribution of Actual vs. Predicted Values')
plt.legend()
plt.show()


# # Final Evaluation

# In[37]:


print("\nTesting the tuned Decision Tree Regressor\n")
y_predict=decisionTree_regressor_tuned.predict(X_test_transformed)
MAE=mean_absolute_error(y_test,y_predict)
MSE=mean_squared_error(y_test,y_predict)
RMSE=np.sqrt(MSE)
R2=r2_score(y_test,y_predict)
print(f"Mean Absolute Error    : {MAE}")
print(f"Mean Squared Error     : {MSE}")
print(f"Root Mean Squared Error: {RMSE}")
print(f"\nR2 Score: {R2}")


# In[38]:


print("\nTesting the tuned Random Forest Regressor\n")
y_predict=randomForest_regressor_tuned.predict(X_test_transformed)
MAE=mean_absolute_error(y_test,y_predict)
MSE=mean_squared_error(y_test,y_predict)
RMSE=np.sqrt(MSE)
R2=r2_score(y_test,y_predict)
print(f"Mean Absolute Error    : {MAE}")
print(f"Mean Squared Error     : {MSE}")
print(f"Root Mean Squared Error: {RMSE}")
print(f"\nR2 Score: {R2}")


# In[39]:


print("\nTesting the tuned xgb Regressor\n")
y_predict=xgb_regressor_tuned.predict(X_test_transformed)
MAE=mean_absolute_error(y_test,y_predict)
MSE=mean_squared_error(y_test,y_predict)
RMSE=np.sqrt(MSE)
R2=r2_score(y_test,y_predict)
print(f"Mean Absolute Error    : {MAE}")
print(f"Mean Squared Error     : {MSE}")
print(f"Root Mean Squared Error: {RMSE}")
print(f"\nR2 Score: {R2}")


# In[ ]:




