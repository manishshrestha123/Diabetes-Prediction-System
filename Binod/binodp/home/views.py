from django.shortcuts import render
from django.http import HttpRequest
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('static/diabetes_dataset.csv')
# Separate features and target column
X = data.drop(columns='Outcome', axis=1)
Y = data['Outcome']
# Data standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = data['Outcome']
# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Load the trained model
SVMmodel = joblib.load('static/svc')
lrmodel = joblib.load('static/LRmodel')

# Load the scaler
scaler = joblib.load('static/scaler.pkl')

def index(request):
    return render(request, 'index.html')

def contact(request):
    return render(request, 'contact.html')

def login(request):
    return render(request, 'login.html')

def registration(request):
    return render(request, 'registration.html')

def faq(request):
    return render(request, 'faq.html')

def prediction(request):
    if request.method == "POST":
        pregnancies = int(request.POST.get('pregnancies'))
        glucose = float(request.POST.get('glucose'))
        bloodPressure = float(request.POST.get('bloodPressure'))
        skinThickness = float(request.POST.get('skinThickness'))
        insulin = float(request.POST.get('insulin'))
        bmi = float(request.POST.get('bmi'))
        diabetesPedigreeFunction = float(request.POST.get('diabetesPedigreeFunction'))
        age = int(request.POST.get('age'))
        model_select = request.POST.get('model_select')

        # Prepare the input data
        input_data = np.array([pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]).reshape(1, -1)

        # Scale the input data using the fitted scaler
        std_data = scaler.transform(input_data)

        # Make prediction using the model
        if model_select == 'SVM':
            prediction = SVMmodel.predict(std_data)
            result = 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'
        elif model_select == 'LR':
            prediction = lrmodel.predict(std_data)
            result = 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'
        elif model_select == 'Both':
            prediction_svm = SVMmodel.predict(std_data)
            result_svm = 'The person is diabetic' if prediction_svm[0] == 1 else 'The person is not diabetic'

            prediction_lr = lrmodel.predict(std_data)
            result_lr = 'The person is diabetic' if prediction_lr[0] == 1 else 'The person is not diabetic'

            result = f'From SVM: {result_svm}. From LR: {result_lr}'

        # Prepare standardized input data for display
        std_pregnancies, std_glucose, std_bloodPressure, std_skinThickness, std_insulin, std_bmi, std_diabetesPedigreeFunction, std_age = std_data[0]

        # Include given input values in the output dictionary
        output = {
            'given_pregnancies': pregnancies,
            'given_glucose': glucose,
            'given_bloodPressure': bloodPressure,
            'given_skinThickness': skinThickness,
            'given_insulin': insulin,
            'given_bmi': bmi,
            'given_diabetesPedigreeFunction': diabetesPedigreeFunction,
            'given_age': age,
            'pred': result,
            'std_pregnancies': std_pregnancies,
            'std_glucose': std_glucose,
            'std_bloodPressure': std_bloodPressure,
            'std_skinThickness': std_skinThickness,
            'std_insulin': std_insulin,
            'std_bmi': std_bmi,
            'std_diabetesPedigreeFunction': std_diabetesPedigreeFunction,
            'std_age': std_age,
        }
        return render(request, 'prediction.html', output)
    else:
        return render(request, 'prediction.html')

    
def about(request):
    return render(request, 'about.html')

def accuracy(request):
    if request.method == 'GET' and 'model_select' in request.GET:
        selected_model = request.GET['model_select']
        if selected_model == 'SVM':
            X_test_prediction_svm = SVMmodel.predict(X_test)
            precision_svm = precision_score(Y_test, X_test_prediction_svm)
            recall_svm = recall_score(Y_test, X_test_prediction_svm)
            f1_svm = f1_score(Y_test, X_test_prediction_svm)
            accuracy_svm = accuracy_score(X_test_prediction_svm, Y_test)
            metrics = {
                'precision': "{:.2f}%".format(precision_svm * 100),
                'recall': "{:.2f}%".format(recall_svm * 100),
                'f1_score': "{:.2f}%".format(f1_svm * 100),
                'accuracy': "{:.2f}%".format(accuracy_svm * 100)
            }
        elif selected_model == 'LR':
            X_test_prediction_lr = lrmodel.predict(X_test)
            precision_lr = precision_score(Y_test, X_test_prediction_lr)
            recall_lr = recall_score(Y_test, X_test_prediction_lr)
            f1_lr = f1_score(Y_test, X_test_prediction_lr)
            accuracy_lr = accuracy_score(X_test_prediction_lr, Y_test)
            metrics = {
                'precision': "{:.2f}%".format(precision_lr * 100),
                'recall': "{:.2f}%".format(recall_lr * 100),
                'f1_score': "{:.2f}%".format(f1_lr * 100),
                'accuracy': "{:.2f}%".format(accuracy_lr * 100)
            }
        else:
            X_test_prediction_svm = SVMmodel.predict(X_test)
            precision_svm = precision_score(Y_test, X_test_prediction_svm)
            recall_svm = recall_score(Y_test, X_test_prediction_svm)
            f1_svm = f1_score(Y_test, X_test_prediction_svm)
            accuracy_svm = accuracy_score(X_test_prediction_svm, Y_test)

            X_test_prediction_lr = lrmodel.predict(X_test)
            precision_lr = precision_score(Y_test, X_test_prediction_lr)
            recall_lr = recall_score(Y_test, X_test_prediction_lr)
            f1_lr = f1_score(Y_test, X_test_prediction_lr)
            accuracy_lr = accuracy_score(X_test_prediction_lr, Y_test)

            metrics = {
                'SVM': {
                    'precision': "{:.2f}%".format(precision_svm * 100),
                    'recall': "{:.2f}%".format(recall_svm * 100),
                    'f1_score': "{:.2f}%".format(f1_svm * 100),
                    'accuracy': "{:.2f}%".format(accuracy_svm * 100)
                },
                'LR': {
                    'precision': "{:.2f}%".format(precision_lr * 100),
                    'recall': "{:.2f}%".format(recall_lr * 100),
                    'f1_score': "{:.2f}%".format(f1_lr * 100),
                    'accuracy': "{:.2f}%".format(accuracy_lr * 100)
                }
            }
        return render(request, 'accuracy.html', {'metrics': metrics, 'selected_model': selected_model})
    else:
        return render(request, 'accuracy.html', {})
