# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 20:26:38 2023

@author: 91974
"""


from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('salary.html')
@app.route('/predict',methods=["POST"])
def predict():
    salary=float(request.values['text'])
    salary=np.reshape(salary,(-1,1))
    output=model.predict(salary)
    output=output.item()
    output=round(output,2)
    #print("hello")
   
    
    return render_template("salary.html",prediction_text="salary is {}".format(output))
if __name__=='__main__':
    app.run(port=8000)
    