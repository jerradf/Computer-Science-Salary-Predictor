from sklearn import svm
import pandas
import numpy
import time
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression



while True:
   
  if __name__ == "__main__":

    print("Welcome to Computer Science Salary Prediction Model!\n")
    year = 0
    while (year < 2000) or (year > 2020):
      try:
        year = float(input("Enter the year (from 2000 to 2020): "))
        if (year < 2000) or (year > 2020):
          print("The year has to be between 2000 and 2020; Try again!")
      except ValueError:
        print("Please type in a year. The year has to be between 2000 and 2020; Try again!")
    y = int(year)

    gender = ""
    while (gender != "Male") and (gender != "Female"):
      gender = input("Enter the gender (Male or Female): ")
      if (gender != "Male") and (gender != "Female"):
        print("Try again!")
    g = gender
    if gender == "Female":
      gender = float(1)
    else:
      gender = float(2)

    occupation = ""
    while (occupation != "Programmer") and (occupation != "Analyst") and (occupation != "Specialist") and (occupation != "Manager"):
      occupation = input("Enter the occupation(Programmer, Analyst, Specialist, Manager): ")
      if (occupation != "Programmer") and (occupation != "Analyst") and (occupation != "Specialist") and (occupation != "Manager"):
        print("Try again!")
    o = occupation
    if occupation == "Programmer":
      occupation = float(0)
    elif occupation == "Analyst":
      occupation = float(1)
    elif occupation == "Specialist":
      occupation = float(2)
    else: # Manager
      occupation = float(3)

    df = pandas.read_csv("computer_science_jobs.csv")
    X = df[["Year", "Gender", "Occupation"]]
    Y = numpy.ravel(df[["WeeklyEarnings"]])



    salary_prediction_model = make_pipeline(PolynomialFeatures(10),LinearRegression())

    salary_prediction_model.fit(X,Y)

# fix prediction, not working properly
    prediction = salary_prediction_model.predict([[year, gender, occupation]])


    print("\n\n\nCalculating the salary for a", g, o, "in", y,"...\n\n\n")


    print("\033[92m" + "------------------ RESULTS ------------------\nThe predicted salary for a", g, o, "in", y, "is", str((prediction[0])*52), "per year.\n\n\n\n")

   

   
    # main program
    while True:
        answer = str(input('\033[0m'+ 'Run again? (yes/no): '))
        if answer in ('yes', 'no'):
            break
        print("invalid input.")
    if answer == 'yes':
        print('\n\n\n')
        continue
    else:
        print("\n\nGoodbye :(")
        break


        