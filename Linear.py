import pandas as pd #pandas is used to read a csv file and extract a array
import numpy as np #can be used to perform mathematical operations on arrays
import sklearn #library full of machine learning algorithms
from sklearn import linear_model #linear models are used to predict in this alggorithm
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot # used to plot the data
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")  #read the csv file using ; as the separater
data = data[["G1", "G2", "G3", "studytime", "failures","absences", "Dalc", "Walc"]] #useful information selected into a array

predict = "G3" # final grade

X = np.array(data.drop([predict], 1))  #X is the array that will be used to predict final grade
y = np.array(data[predict]) #used to predict final grade(label)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size= 0.2) #split into 4 different arrays, 2 for training and 2 for testing. testing size 20%
linear = linear_model.LinearRegression() #defining the type of regression we will use

linear.fit(x_train, y_train) #finds a best fit line and stores it in linear

acc = linear.score(x_test,y_test) #tests the model using the testing data
print(acc) #print the accuracy

"""with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f) #save a pickle file for us in the directory 
    
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in) """#save the model so you dont need to keep developing the model everytime(Take a second look at this)

print('Co:  \n', linear.coef_) #prints the coefficients of the linear lines
print('intercept \n', linear.intercept_) #prints the y intercept

predictions = linear.predict(x_test) #prediction of every row in the array

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x]) #print all the predictions, test data and real grade

p = 'G1' #define p as G1 array
style.use("ggplot") #style of sheet
pyplot.scatter(data[p], data["G3"]) #use a scatter plot, define the plot data
pyplot.xlabel(p) #xlabel
pyplot.ylabel("Final Grade") #ylabel
pyplot.show() #show the plot