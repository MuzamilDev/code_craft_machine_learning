#Linear Regression Model for Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #machine learning model
from sklearn.metrics import mean_squared_error, r2_score

data =pd.read_csv('/content/train.csv')
ind_reg = ['GrLivArea','BedroomAbvGr','FullBath']
dep_reg =['SalePrice']
X= data[ind_reg] #independent variable
Y= data[dep_reg] #dependent varaible

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
Linear_model= LinearRegression()
Linear_model.fit(X_train,Y_train)

y_pred = Linear_model.predict(X_test)
mse = mean_squared_error(Y_test,y_pred)
r2 = r2_score(Y_test,y_pred)

#Priniting metrics values
#print(f"Mean Squared Error (MSE): {mse:,.2f}")
#print(f"R-squared (R^2): {r2:.2f}")

new_data_house = pd.DataFrame([[1700,6,6]], columns=ind_reg) #columns features
p_price = Linear_model.predict(new_data_house)

print("\n Pridiction for new house:")
print(f"features: Square Footage=1500, Bedroom=3, Bathrooms=2")
print(f"Predicted price: $ {p_price[0][0]:,.2f}")

