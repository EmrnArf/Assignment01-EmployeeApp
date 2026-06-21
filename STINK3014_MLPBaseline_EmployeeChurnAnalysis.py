# This is a baseline code for the Employee Churn Modeling
# You are required to ammend this code based on related questions 

#calling respectives libraries to be used
import numpy as np
# Import MLPClassifer 
from sklearn.neural_network import MLPClassifier
# Import accuracy score 
from sklearn.metrics import accuracy_score
# Import train_test_split function
from sklearn.model_selection import train_test_split
# This part cover the neural network implementation of this problem
from sklearn import preprocessing
# Import pandas library for database manipulation
import pandas as pd
# Import matplotlib & seaborn for visual dislays
import matplotlib.pyplot as plt
import seaborn as sns
from os import system

# Load database (using Panda function)
# This dataset can be downloaded from Kaggle website - file name 'HR_comma_sep.csv
# database format = comma-separated value
# please put this file in the same folder of your python code
data=pd.read_csv('HR_comma_sep.csv')

# This part cover the visualization parts of the dataset
# data description
print(data.head())
print(data.info())
print(data.describe(include ='all'))



#multiple displays of the dataset
features=['number_project','time_spend_company','Work_accident','left','promotion_last_5years','Departments','salary']
fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
     plt.subplot(4, 2, i+1)
     plt.subplots_adjust(hspace = 1.0)
     sns.countplot(x=j,data = data)
     plt.xticks(rotation=90)
     plt.title("No. of employee")
plt.show()

#correlation mapping - using a heatmap visual diagram
correlation = data.corr(numeric_only=True)
plt.figure(figsize=(10,10))
sns.heatmap(correlation,vmax=1,vmin=-1,square=True,annot=True,linewidths=.5,cmap="YlGnBu")
plt.show()

# --- 3D Scatterplot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Plotting Satisfaction, Last Evaluation, and Average Monthly Hours, color-coded by if they 'left'
scatter = ax.scatter(data['satisfaction_level'], data['last_evaluation'], data['average_montly_hours'], c=data['left'], cmap='coolwarm', alpha=0.5)
ax.set_xlabel('Satisfaction Level')
ax.set_ylabel('Last Evaluation')
ax.set_zlabel('Avg Monthly Hours')
plt.title("3D Scatterplot: Satisfaction, Evaluation, and Hours vs Churn (Red=Left)")
plt.show()

# --- Box Plots ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='left', y='satisfaction_level', data=data, palette='Set2')
plt.title('Box Plot: Satisfaction Level by Churn')

plt.subplot(1, 2, 2)
sns.boxplot(x='left', y='average_montly_hours', data=data, palette='Set2')
plt.title('Box Plot: Avg Monthly Hours by Churn')
plt.tight_layout()
plt.show()

#  Import LabelEncoder & creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers (ordinal scale).
# - original values in string
data['salary']=le.fit_transform(data['salary'])
data['Departments']=le.fit_transform(data['Departments'])




# Spliting data into Feature (X) and target (T)
# X represent features or variables
X=data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 
     'Work_accident','promotion_last_5years', 'Departments', 'salary']]
#Y represents target
y=data['left']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% test

# Create model object for a neural network classifier 
clf = MLPClassifier(hidden_layer_sizes=(8,5), max_iter=2000, activation = 'relu',
                    solver='adam', random_state=5, tol=0.001, momentum=0.9,
                    verbose=True, learning_rate_init=0.01, early_stopping=False, n_iter_no_change=500)

# Fit data onto the model
clf.fit(X_train,y_train)

# Make prediction on test dataset
ypred=clf.predict(X_test)


# Calcuate accuracy
print('\nClassification Accuracy: %.3f%%' %(accuracy_score(y_test,ypred)*100))

#plotting the error / loss curve
plt.plot(clf.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.ylim(0,1)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

#function to clear screen
cls = lambda: system('cls')
cls()

#ask new inputs from users
print('\n++++++++++++++++++++++++++++++++++')
print('Employee Churn Analytics System')
print('++++++++++++++++++++++++++++++++++\n')
print('Enter new inputs')
staf_name = str(input('Enter your staf name:'))
satisfaction_level = float(input('Employee Satisfaction (0-1): '))
last_evaluation = float(input('Evaluation grade (0-1): '))
number_project = int(input('Project involvement: '))
average_montly_hours =int(input('Average Monthly Hours: '))
time_spend_company = int(input('Time Spend : '))
Work_accident = int(input('Work Accident (0, 1): '))
promotion_last_5years = int(input('Promotion last 5-years (0,1): '))
Departments = int(input('Department (1-10) : '))
salary = int(input('Salary (1-3): '))

# function to assign new inputs to be used in the model
new_input = np.array([satisfaction_level, last_evaluation, number_project, 
                      average_montly_hours, time_spend_company, Work_accident,
                      promotion_last_5years, Departments, salary])
predictNew = clf.predict(new_input.reshape(1,-1))

# show the predicted results
print('\n+++++++++++++++++++++++++++++++')
print('Classification Result :')
print('+++++++++++++++++++++++++++++++')
if predictNew == [1]:
     print('Predicted Result: >> Sorry, %s tends to left the company.\n' %staf_name)
else:
     print('Predicted Result: >> Good news! Most likely %s will stay.\n' %staf_name)

print('(This predicted result is based on the accuracy rate: %.3f%% after %dth iteration with the lowest error %.3f)\n' 
          %(accuracy_score(y_test,ypred)*100, clf.n_iter_, clf.best_loss_))
