from sklearn import svm
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


df = pandas.read_csv("computer_science_jobs.csv")
X = df[["Year", "Gender", "Occupation"]]
Y = numpy.ravel(df[["WeeklyEarnings"]])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)


# Exp 1-1
print('Exp 1-1\n')
# SVM - SVC
salary_prediction_model = svm.SVC(kernel="linear", C=.99)
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('SVC:  ',result) # accuracy % 


# SVM - SVR
salary_prediction_model = svm.SVR(kernel="linear", C=.99)
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('SVR:  ',result) # accuracy % 


# Linear Regression 
salary_prediction_model = linear_model.LinearRegression()
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('LR:  ', result) # accuracy

#LASSOCV
salary_prediction_model = linear_model.LassoCV(cv=5)
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('LassoCV cv=5:  ', result)

#LARS LASSO Algorithm
salary_prediction_model = linear_model.LassoLars(alpha=.1)
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('LassoLARS a=.1:  ', result)

#LassoLARSCV
salary_prediction_model = linear_model.LassoLarsCV(cv=5)
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('LassoLARSCV cv=5:  ', result)


# Polynomial Regression - 2 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(2),LinearRegression())
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('PR(2):  ', result) # accuracy

#Gaussian Process Regression
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,random_state=0)
gpr.fit(X_train, y_train)
result = gpr.score(X_test, y_test)
print('GPR:  ', result) # accuracy

#--------------------------------------------------
print('\n\nExp 1-2\n')
# Exp 1-2
# Polynomial Regression - 3 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(3),LinearRegression())
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('PR(3):  ', result) # accuracy

# Polynomial Regression - 5 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(5),LinearRegression())
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('PR(5):  ', result) # accuracy

# Polynomial Regression - 10 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(10),LinearRegression())
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('PR(10):  ', result) # accuracy

# Polynomial Regression - 15 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(15),LinearRegression())
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('PR(15):  ', result) # accuracy

# Polynomial Regression - 30 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(30),LinearRegression())
salary_prediction_model.fit(X_train, y_train)

result = salary_prediction_model.score(X_test, y_test)
print('PR(30):  ', result) # accuracy
#---------------------------------------------------------

# EXPERIMENT 1-3

print('\n\n\nExp 1-3\n\ntest size 0.4:')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)


# SVM - SVC
salary_prediction_model = svm.SVC(kernel="linear", C=.99)
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('SVC:  ',result) # accuracy % 

# SVM - SVR
salary_prediction_model = svm.SVR(kernel="linear", C=.99)
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('SVR:  ',result) # accuracy % 

# Linear Regression 
salary_prediction_model = linear_model.LinearRegression()
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('LR:  ', result) # accuracy

# Polynomial Regression - 2 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(2),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(2):  ', result) # accuracy

# Exp 1-2
# Polynomial Regression - 3 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(3),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(3):  ', result) # accuracy

# Polynomial Regression - 5 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(5),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(5):  ', result) # accuracy

# Polynomial Regression - 10 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(10),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(10):  ', result) # accuracy
# Polynomial Regression - 15 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(15),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(15):  ', result) # accuracy
# Polynomial Regression - 30 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(30),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(30):  ', result) # accuracy

salary_prediction_model = make_pipeline(PolynomialFeatures(50),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(50):  ', result) # accuracy

#---------------------------------------------------------

print('\ntest size 0.1:')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)


# SVM - SVC
salary_prediction_model = svm.SVC(kernel="linear", C=.99)
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('SVC:  ',result) # accuracy % 

# SVM - SVR
salary_prediction_model = svm.SVR(kernel="linear", C=.99)
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('SVR:  ',result) # accuracy % 

# Linear Regression 
salary_prediction_model = linear_model.LinearRegression()
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('LR:  ', result) # accuracy

# Polynomial Regression - 2 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(2),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(2):  ', result) # accuracy

# Polynomial Regression - 3 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(3),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(3):  ', result) # accuracy

# Polynomial Regression - 5 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(5),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(5):  ', result) # accuracy

# Polynomial Regression - 10 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(10),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(10):  ', result) # accuracy

# Polynomial Regression - degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(15),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(15):  ', result) # accuracy

# Polynomial Regression - 30 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(30),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(30):  ', result) # accuracy

#---------------------------------------------------------

print('\n\ntest size 0.05:')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=0)

# SVM - SVC
salary_prediction_model = svm.SVC(kernel="linear", C=.99)
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('SVC:  ',result) # accuracy 

salary_prediction_model = svm.SVR(kernel="linear", C=.99)
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('SVR:  ',result) # accuracy 

# Linear Regression 
salary_prediction_model = linear_model.LinearRegression()
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('LR:  ', result) # accuracy

# Polynomial Regression - 2 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(2),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(2):  ', result) # accuracy

# Polynomial Regression - 3 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(3),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(3):  ', result) # accuracy

# Polynomial Regression - 5 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(5),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(5):  ', result) # accuracy

# Polynomial Regression - 10 degrees
salary_prediction_model = make_pipeline(PolynomialFeatures(10),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(10):  ', result) # accuracy

salary_prediction_model = make_pipeline(PolynomialFeatures(15),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(15):  ', result) # accuracy

salary_prediction_model = make_pipeline(PolynomialFeatures(30),LinearRegression())
salary_prediction_model.fit(X_train, y_train)
result = salary_prediction_model.score(X_test, y_test)
print('PR(30):  ', result) # accuracy
#-----------------------------------------------------------




# Research: How Accurate is the ML/Prediction?
# 1-1: Which ML algorithms is the best?
# SVM, Linear R, Poly R, Another???
# bar chart 

# 1-2: For the best algorithm, what is the best config?
# 

# 1-3: How the test split will affect the accuracy?
# 0.2, 0.1, 0.05, 0.4


# ?1-4: How is each feature affecting (weight) the final result?
# 