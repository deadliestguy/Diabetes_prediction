from flask import Flask, request , render_template , jsonify

import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import pickle ## for dump the model 

application = Flask(__name__)
app = application

# model guloke load korte hbe
scaler = pickle.load(open('models/scaler.pkl','rb'))
regressor = pickle.load(open('models/regressor.pkl' ,'rb'))
dtc = pickle.load(open('models/dtc.pkl','rb'))
svc = pickle.load(open('models/svc.pkl','rb'))
gnb = pickle.load(open('models/gnb.pkl','rb'))


@app.route("/")
def home_page():
    return render_template('home.html')

# prediction er jonno function 
@app.route('/predict' , methods=['GET','POST'])
def predict_datapoint():

    if request.method == 'POST':
        #store all the input 
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = int(request.form.get('Age'))

        # scale down all input 
        scaled_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

        # predict teh output 
        pred = regressor.predict(scaled_data)

        if pred == 0:
            results = 'Non diabatic'
        else:
            results = 'Diabatic'

        # show the result 
        return render_template('single_prediction.html' , result = results)

    
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
