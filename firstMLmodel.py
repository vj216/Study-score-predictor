#STEP 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#STEP 2: Prepare the data
data = {
    'Hours_Studied': [1,2,3,4,5,6,7,8,9,10],
    'Score': [12,25,32,40,50,55,65,72,80,90]
}

df = pd.DataFrame(data)
 
#Goal of the model is to teach the computer to predict the score a student will get based on the number of hours they have studied

#STEP 3: Split the data into training and testing sets
X = df[['Hours_Studied']]
y = df[['Score']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
#training set teaches the model and testing set tells us how well it did

#STEP 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

#STEP 5: Make predictions
y_pred = model.predict(X_test)
print("Predictions: ", y_pred)

#STEP 6: Evaluate the model
#This tells us how far we were from the predicted result
mse = mean_squared_error(y_test, y_pred)
print("Mean squared Error: ",mse)

print('Predicted score for a student, who studied for 5 hours is',model.predict(pd.DataFrame({'Hours_Studied':[5]})))
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)



#STEP 7: Visualize results
#Convert X to 1D array for plotting
X_1d = X['Hours_Studied'].values
y_pred_full = model.predict(X).flatten()

plt.scatter(X_1d, y, color='purple', label='Actual Scores')
plt.plot(X_1d, y_pred_full, color="orange", label="Predicted Line" )
plt.xlabel('Hours Studied')
plt.ylabel('Hours Studied vs Score Prediction')
plt.legend()
plt.show()
