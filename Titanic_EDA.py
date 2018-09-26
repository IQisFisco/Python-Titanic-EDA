# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## what packages will i need
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as matplt
import matplotlib.patches as matpat

#importing titanic dataset
file_loc = "C:/Users/IEUser/Documents/DataScience-AI-master/Labs/dat/"
file_name = "titanic.csv"
titanic = pd.read_csv(file_loc + file_name)

sns.set()
sns.set_style("ticks")

# setting pandas display
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_rows', None)

# sampling the data
print(titanic.sample(15))

# describing the dataset
titanic.describe() #stats for numeric values
titanic.describe(include=['O'])

# simple plotting of survivals againts non survivals
lived = sum(titanic["Survived"]==1)
died = sum(titanic["Survived"]==0)
x = np.arange(2)
red_patch = matpat.Patch(color = "red", label="Died")
green_patch = matpat.Patch(color = "green", label="Lived")
matplt.legend(handles = [red_patch, green_patch], loc = "upper left")
matplt.xticks(x, list(["Lived", "Died"]))
matplt.ylabel("No of Passengers ")
matplt.bar(x, [lived, 0], color = "green")
matplt.bar(x, [0, died], color = "red")
matplt.show()

# understanding the impact of fare
_ = sns.swarmplot(x = titanic["Survived"], y = titanic["Fare"], order = [0,1], data = titanic)
matplt.xticks(x, list(["Lived", "Died"]))
matplt.show()

# creating cumulative density function
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1)/n
    return x, y

# using the ecdf
lived_Fare = np.array(list(titanic["Fare"][titanic["Survived"]==1]))
died_Fare = np.array(list(titanic["Fare"][titanic["Survived"]==0]))

lived_Fare_x, lived_Fare_y = ecdf(lived_Fare)
died_Fare_x, died_Fare_y = ecdf(died_Fare)
_ = matplt.plot(lived_Fare_x, lived_Fare_y, linestyle = "none", marker = ".")
_ = matplt.plot(died_Fare_x, died_Fare_y, linestyle = "none", marker = "+")
_ = matplt.legend(("Lived","Died"), loc = "upper left")
_ = matplt.margins(0.02)
_ = matplt.xlabel("Fares")
_ = matplt.ylabel("ECDF")
matplt.show()

# summarizing people in each class
print(titanic[["Pclass","Fare"]].groupby("Pclass").agg(["count", "min", "max"]))

# creating the "lived" and "Died" counts per passenger class
survivalPClass = []
for ind_Pclass, val_Pclass in enumerate(titanic.Pclass.unique()):
    for ind_Survived, val_Survived in enumerate(titanic.Survived.unique()):
        count = titanic[["Survived","Pclass"]][np.logical_and(titanic["Pclass"] == val_Pclass,
                        titanic["Survived"] == val_Survived)].Survived.count()
        survivalPClass.append({
                "PclassAgg" : val_Pclass,
                "SurvivedAgg" : val_Survived,
                "Count" : count
                })
aggData = pd.DataFrame(survivalPClass)
aggDataSorted = aggData.sort_values("PclassAgg")
print(aggDataSorted)

# plotting the "lived" and "Died" counts per passenger class
lived = list(aggDataSorted["Count"][aggDataSorted["SurvivedAgg"] == 1])
died = list(aggDataSorted["Count"][aggDataSorted["SurvivedAgg"] == 0])
x = np.arange(len(lived))
width = 0.27
red_patch = matpat.Patch(color = "red", label="Died")
green_patch = matpat.Patch(color = "green", label="Lived")
matplt.legend(handles = [red_patch, green_patch], loc = "upper left")
_ = matplt.xticks(x, list(aggDataSorted["PclassAgg"].unique()), )
_ = matplt.bar(x, lived, width = width, color = "green")
_ = matplt.bar(x + width, died, width = width, color = "red")
_ = matplt.ylabel("counts")
_ = matplt.xlabel("Passenger Class")
matplt.show()

# Dealing with Age
# Idea is to assign ages based on initials
# first extracting initials
titanic["initials"] = 0
for i in titanic:
    titanic["initials"]= titanic.Name.str.extract('([A-Za-z]+)\.')

titanic["initials"].replace(["Don", "Rev", "Dr", "Mme", "Ms", "Major", "Lady",
                       "Sir", "Mlle", "Col", "Capt", "Countess", "Jonkheer"],
                            ["Mr", "Mr", "Mr", "Miss", "Miss", "Other", "Mrs",
                        "Mr", "Miss", "Other", "Other", "Other", "Other"], 
                        inplace = True)

print(titanic.groupby("initials")["Age"].mean())

titanic.loc[(titanic.Age.isnull()) & (titanic.initials == "Master"), "Age"] = 5
titanic.loc[(titanic.Age.isnull()) & (titanic.initials == "Miss"), "Age"] = 22
titanic.loc[(titanic.Age.isnull()) & (titanic.initials == "Mr"), "Age"] = 33
titanic.loc[(titanic.Age.isnull()) & (titanic.initials == "Mrs"), "Age"] = 36
titanic.loc[(titanic.Age.isnull()) & (titanic.initials == "Mrs"), "Age"] = 51

_ = matplt.xlabel('Age')
_ = matplt.ylabel('No. of Passengers')
bins = int(np.round(np.sqrt(len(titanic["Age"]))))
_ = matplt.hist(titanic["Age"], bins= bins)
matplt.show()

# Checking age impact on "Lived" or "Died"
lived = titanic["Age"][titanic["Survived"] == 1]
died = titanic["Age"][titanic["Survived"] == 0]

lived_x, lived_y = ecdf(lived)
died_x, died_y = ecdf(died)

_ = matplt.plot(lived_x, lived_y, linestyle="none", marker = ".")
_ = matplt.plot(died_x, died_y, linestyle="none", marker = ".")


_ = matplt.margins(0.1)

matplt.legend(("Lived", "Died"), loc="lower right")
_ = matplt.xlabel("Age")
_ = matplt.ylabel("ECDF")

matplt.show()

# scatter plotting
g = sns.FacetGrid(titanic, hue="Survived", col="Sex", margin_titles=True,
                palette="Set1",hue_kws=dict(marker=["^", "v"]))
g.map(matplt.scatter, "Fare", "Age",edgecolor="w").add_legend()
matplt.subplots_adjust(top=0.8)
g.fig.suptitle('Survival by Gender , Age and Fare')
matplt.show()