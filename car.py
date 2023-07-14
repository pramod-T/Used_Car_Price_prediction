# example of training to the test set for the housing dataset
from pandas import read_csv
import pandas as pd
import numpy as np 
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# load the dataset
train = read_csv("train-data.csv")


print(type(train))
#print(train.isnull().sum())
#removing electric vehicles
train = train[train['Fuel_Type'] != 'Electric']


#print(train.Fuel_Type.value_counts())
#print(train.Kilometers_Driven.min())
#print(train.Kilometers_Driven.max())

#Removing the outliers in Kilomerers_Driven
train = train[train['Kilometers_Driven'] < 500000]
train = train[train['Kilometers_Driven'] > 1000]

#print(train.shape)

#Modifying car names to group by brand
train.Name = train.Name.str.split().str.get(0)


#print(train.head())
#print(train.Name.value_counts())

#Removing outliers in car brands
train = train[train['Name'] != 'Force']
train = train[train['Name'] != 'ISUZU']
train = train[train['Name'] != 'Bentley']
train = train[train['Name'] != 'Lamborghini']
train = train[train['Name'] != 'Isuzu']
train = train[train['Name'] != 'Smart']
train = train[train['Name'] != 'Ambassador']

#print(train['Name'].unique())
#print(train['Location'].unique())

#print(train.Name.value_counts())

#Removing Outliers in Price
#sns.boxplot(train.Price)
#print(train.Price.min())
#print(train.Price.max())
train = train[train.Price < 120]
train = train[train.Price > 0.5]


#Converting Mileage, Engine and Power to numerical columns
train.Mileage = train.Mileage.str.split().str.get(0).astype('float')
train.Engine = train.Engine.str.split().str.get(0).astype('int', errors='ignore')
train.Power = train.Power.str.split().str.get(0).astype('float', errors='ignore')
print(train.head())


#finding age

train['Car_age'] = 2021 - train['Year']
#train.head()

#train.Price = np.log1p(train.Price)

#Performing label encoding for categorical data
train['Name'] = label_encoder.fit_transform(train['Name'])
train['Location'] = label_encoder.fit_transform(train['Location'])
train['Fuel_Type'] = label_encoder.fit_transform(train['Fuel_Type'])
train['Transmission'] = label_encoder.fit_transform(train['Transmission'])
train['Owner_Type'] = label_encoder.fit_transform(train['Owner_Type'])


#print(train.head())
#Dealing with missing values
#print(train.isnull().sum())
#print(train.dtypes)
train.Engine = pd.to_numeric(train.Engine, errors='coerce')
train.Power = pd.to_numeric(train.Power, errors='coerce')

#print(train.dtypes)

imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
train[["Engine", "Power", "Seats"]] = imputer.fit_transform(train[["Engine", "Power", "Seats"]])

#print(train['Name'].unique())
#print(train.head())
#print(train.Name.max())
#print(train.dtypes)
#print(train.isnull().sum())
print(train.head())
y = train.Price
x = train.drop(['Price'],axis=1)

# split dataset
X_train, X_valid, Y_train, Y_valid = train_test_split(x,y,test_size=0.2)

#tain the model
#model = LinearRegression() 
#model.fit(X_train, Y_train) 
#y_pred = model.predict(X_valid)

# evaluate predictions
#mae = mean_absolute_error(Y_valid, y_pred)
#print('MAE: %.3f' % mae)

#train the model
model1 = RandomForestRegressor()
model1.fit(X_train, Y_train)
#y_pred = model1.predict(X_valid)



#evaluate predictions
#mae = mean_absolute_error(Y_valid, y_pred)
#print('MAE: %.3f' % mae)


#Save Model        
import pickle
file = open('car.pkl', 'wb')
pickle.dump(model1, file)
file.close()









