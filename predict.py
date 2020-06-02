import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def convert_to_int(num):
    if '$ ' in num:
        num=num.replace('$ ','')
    num=int(num)
    return num

budget=[]
sales=[]
res_to_predict_of_sales = []

data = pd.read_csv('input.csv')

for i in range(len(data['Sales'].values)):
    a=str(data['Sales'].values[i])
    b=str(data['Budget'].values[i])
    if 'nan' in a:
        break

    vala = convert_to_int(a)
    valb = convert_to_int(b)

    budget.append(valb)
    sales.append(vala)

budget = np.array(budget).reshape(-1,1)
sales = np.array(sales).reshape(-1,1)

for i in range(len(sales), len(data['Budget'].values)):
    val = str(data['Budget'].values[i])
    val = convert_to_int(val)
    res_to_predict_of_sales.append(val)

# print(sales)
# res_to_predict_of_sales = np.array(res_to_predict_of_sales).reshape(-1,1)
# print(res_to_predict_of_sales)

# # Model initialization
regression_model = LinearRegression()

regression_model.fit(budget, sales)
# Predict
sales_predicted = regression_model.predict(budget)

rmse = mean_squared_error(sales, sales_predicted)
r2 = r2_score(sales, sales_predicted)

print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

#Now, predict the values for the first quarter of 2019
slope = regression_model.coef_[0][0]
intercept = regression_model.intercept_[0]

jan_2019_budget = res_to_predict_of_sales[0]
feb_2019_budget = res_to_predict_of_sales[1]
march_2019_budget = res_to_predict_of_sales[2]

jan_2019_sales_prediction = slope*jan_2019_budget + intercept
feb_2019_sales_prediction = slope*feb_2019_budget + intercept
march_2019_sales_prediction = slope*march_2019_budget + intercept

def check_goal(x,y):
    print(x,y)
    if(x<y):
        return 'no'
    else: return 'yes'

print()
print("**************Predicted result for First quarter of 2019***************")
print()
print("I am assuming if 'sale < budget' than company won't hit their goal")

print()
print("jan 2019 sale", jan_2019_sales_prediction)
print("Will company hit their goal ?", check_goal(jan_2019_sales_prediction,jan_2019_budget))
print()

print("feb 2019 sale", feb_2019_sales_prediction)
print("Will company hit their goal ?", check_goal(feb_2019_sales_prediction,feb_2019_budget))
print()

print("march 2019 sale", march_2019_sales_prediction)
print("Will company hit their goal ?", check_goal(march_2019_sales_prediction,march_2019_budget))


# plotting values

plt.scatter(budget, sales, s=10)
plt.xlabel('Budget')
plt.ylabel('Sales')

# predicted values
plt.plot(budget, sales_predicted, color='r')
plt.show()