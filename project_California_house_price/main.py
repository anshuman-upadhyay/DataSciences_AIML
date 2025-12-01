from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score



import pandas as pd
import numpy as np

#1.Loading dataset
housing=pd.read_csv("housing.csv")

#2 Creating a stratified test split
housing["income_cat"]= pd.cut(housing["median_income"],bins=[0.0,1.5,3.0,4.5,6.0 ,np.inf],labels=[1,2,3,4,5])

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=12)
for train_index,test_index in split.split(housing,housing['income_cat']):
    strat_train_set=housing.loc[train_index].drop("income_cat",axis=1)
    strat_test_set=housing.loc[test_index].drop("income_cat",axis=1)

# We will work on a copy of the training data
housing=strat_train_set.copy()
#3 separate features and labels 
housing_lables= housing["median_house_value"].copy()
housing=housing.drop("median_house_value",axis=1)

# print(housing,housing_lables)

#4. Separate numerical and categorical columns
num_attirbs=housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs= ["ocean_proximity"]

#5. Pipeline Construction

#Making pipeline for numerical data 
num_pipe= Pipeline([
    ("imputer", SimpleImputer()),
    ("scalar", StandardScaler()),
])
#Making pipeline for categorical data 
cat_pipe= Pipeline([
    ("encoder" , OneHotEncoder(handle_unknown="ignore")),
])

#Complete pipeline construction

full_pipeline= ColumnTransformer([
    ("num", num_pipe,num_attirbs),
    ("cat", cat_pipe,cat_attribs)
])

#6 Transform Data
housing_prepared=full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

#7.  Train the model
#Linear Regression model
lr=LinearRegression()
lr.fit(housing_prepared,housing_lables)
l_pred=lr.predict(housing_prepared)
lr_rmse=root_mean_squared_error(housing_lables,l_pred)
print(f"The Root Mean Squared Error for Linear Regression is {lr_rmse}")
lin_rmses= - cross_val_score(lr,housing_prepared,housing_lables,scoring="neg_root_mean_squared_error",cv=10)
print(f"The Mean of all the Validation in Linear Regression is : {pd.Series(lin_rmses).mean()}")

#RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(housing_prepared,housing_lables)
rfr_pred=rfr.predict(housing_prepared)
rfr_rmse=root_mean_squared_error(housing_lables,rfr_pred)
print(f"The Root Mean Squared Error for Random Forest Regressor is {rfr_rmse}")
rfr_rmses= - cross_val_score(rfr,housing_prepared,housing_lables,scoring="neg_root_mean_squared_error",cv=10)
print(f"The Mean of all the Validation in Random Forest Regressor is : {pd.Series(rfr_rmses).mean()}")


#DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(housing_prepared,housing_lables)
dtr_pred=dtr.predict(housing_prepared)
dtr_rmse=root_mean_squared_error(housing_lables,dtr_pred)
print(f"The Root Mean Squared Error for Decision Tree Regressor is {dtr_rmse}")
tree_rmses= - cross_val_score(dtr,housing_prepared,housing_lables,scoring="neg_root_mean_squared_error",cv=10)
print(f"The Mean of all the Validation in Decision Tree Regressor is : {pd.Series(tree_rmses).mean()}")
# print(f"The Negative Root Mean Squared Error for Decision Tree Regressor is {}")






