
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#STEP 1: DATA READING

df = pd.read_csv("images_analyzed_productivity1.csv")
print(df.head())
plt.figure(1)
plt.scatter(df.Age, df.Productivity, marker='+', color='red')
plt.ylabel("Productivity")
plt.xlabel("Age")
#plt.show()
plt.figure(2)
plt.scatter(df.Time, df.Productivity, marker='+', color='red')
plt.ylabel("Productivity")
plt.xlabel("Time")
#plt.show()
plt.figure(3)
plt.scatter(df.Coffee, df.Productivity, marker='+', color='red')
plt.ylabel("Productivity")
plt.xlabel("Coffee")
#plt.show()

#PLot productivity values to see the split between Good and Bad
sizes = df['Productivity'].value_counts(sort = 1)

plt.figure(4)
plt.pie(sizes, shadow=True, autopct='%1.1f%%')
#plt.show()
#Good to know so we know the proportion of each label


#STEP 2: DROP IRRELEVANT DATA
#In our example, Images_Analyzed reflects whether it is good analysis or bad
#so should not include it. ALso, User number is just a number and has no inflence
#on the productivity, so we can drop it.

df.drop(['Images_Analyzed'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)


#STEP 3: Handle missing values, if needed
#df = df.dropna()  #Drops all rows with at least one null value.


#STEP 4: Convert non-numeric to numeric, if needed.
#Sometimes we may have non-numeric data, for example batch name, user name, city name, etc.
#e.g. if data is in the form of YES and NO then convert to 1 and 2

df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 2
print(df.head())


#STEP 5: PREPARE THE DATA.

#Y is the data with dependent variable, this is the Productivity column
Y = df["Productivity"].values  #At this point Y is an object not of type int
#Convert Y to int
Y=Y.astype('int')

#X is data with independent variables, everything except Productivity column
# Drop label column from X as you don't want that included as one of the features
X = df.drop(labels = ["Productivity"], axis=1)
#print(X.head())

#STEP 6: SPLIT THE DATA into TRAIN AND TEST data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)
#random_state can be any integer and it is used as a seed to randomly split dataset.
#By doing this we work with same test dataset evry time, if this is important.
#random_state=None splits dataset randomly every time

#print(X_train)


from sklearn.linear_model import LogisticRegression   #Import the relevant model
model = LogisticRegression()  #Create an instance of the model.

model.fit(X_train, y_train)  # Train the model using training data



prediction_test = model.predict(X_test)

from sklearn import metrics
#Print the prediction accuracy
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
#Test accuracy for various test sizes and see how it gets better with more training data

'''
#UNDERSTAND WHICH VARIABLES HAVE MOST INFLUENCE ON THE OUTCOME
# To get the weights of all the variables

print(model.coef_) #Print the coefficients for each independent variable.
#But it is not clear which one corresponds to what.
#SO let us print both column values and coefficients.
#.Series is a 1-D labeled array capable of holding any data type.
#Default index would be 0,1,2,3... but let us overwrite them with column names for X (independent variables)
weights = pd.Series(model.coef_[0], index=X.columns.values)

print("Weights for each variables is a follows...")
print(weights)

#+VE VALUE INDICATES THAT THE VARIABLE HAS A POSITIVE IMPACT'''

baseline = model.score(X_test,y_test)

import diffprivlib.models as dp
dp_clf = dp.LogisticRegression()
dp_clf.fit(X_train, y_train)

print("Differentially private test accuracy (epsilon=%.2f): %.2f%%" %
     (dp_clf.epsilon, dp_clf.score(X_test, y_test) * 100))

dp_clf = dp.LogisticRegression(epsilon=float("inf"), data_norm=1e5)
dp_clf.fit(X_train, y_train)

print("Agreement between non-private and differentially private (epsilon=inf) classifiers: %.2f%%" % (dp_clf.score(X_test, model.predict(X_test)) * 100))

accuracy = []
epsilons = np.logspace(-3, 1, 500)

for eps in epsilons:
    dp_clf = dp.LogisticRegression(epsilon=eps, data_norm=100)
    dp_clf.fit(X_train, y_train)
    accuracy.append(dp_clf.score(X_test, y_test))

import pickle

pickle.dump((epsilons, baseline, accuracy), open("lr_accuracy_500.p", "wb" ) )

epsilons, baseline, accuracy = pickle.load(open("lr_accuracy_500.p", "rb"))

plt.figure(6)
plt.semilogx(epsilons, accuracy, label="Differentially private")
plt.plot(epsilons, np.ones_like(epsilons) * baseline, dashes=[2,2], label="Non-private")
plt.title("Differentially private logistic regression accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xlim(epsilons[0], epsilons[-1])
plt.legend(loc=3)
plt.show()