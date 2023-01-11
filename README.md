# PfDAProject2

### Project 2 in Programming for Data Analytics module.

G00411297 Sadie Concannon

This project investigates the Wisconsin Breast Cancer dataset

### Problem: 
The following list presents the requirements of the project

- Undertake an analysis/review of the dataset and present an overview and background.
- Provide a literature review on classifiers which have been applied to the dataset and compare their performance
- Present a statistical analysis of the dataset
- Using a range of machine learning algorithms, train a set of classifiers on the dataset (using SKLearn etc.) and present   classification performance results. 
- Compare, contrast and critique your results with reference to the literature
- Discuss and investigate how the dataset could be extended – using data synthesis of new tumour datapoints
- Document your work in a Jupyter notebook.
- Please use GitHub to demonstrate research, progress and consistency

#### Table of Contents
- General Info
- Technologies
- Setup

#### General Info
This repository is created to complete Project 2 for the Programming for Data Analytics module as part of the Higher Diploma in Data Analytics in Computing.

This repository contains one Jupyter notebook that contains the main findings and all statistical analysis and machine learning coding. It also contains a jpeg image and references bibtex file.

#### Technologies
To run the noteboook on your local machine you will need Pyhton 3.9 or above. This notebook has been completed using Anaconda and the following imported the following libraries;

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #for plotting
import sklearn as sk
import seaborn as sns
import plotly.express as px
import warnings #to ignore filter warnings
#for machine learning
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.exceptions import FitFailedWarning
'''

#### Setup
To run this notebook on a local machine you should do the following:

Open the command line on your machine. I use terminal in Visual Studio code but also recommend CMDer if working on a Windows device.
Navigate to the directory that contains the downloaded notebook
In the terminal type "Jupyter Notebook" and press "Enter"
This will open the notebook on the Jupyter browser
For this project I used visual studio code for ease of pushing to GitHub.
