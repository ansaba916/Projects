# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:27:18 2023

@author: 91974
"""

from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('prediction.html')
@app.route('/predict',methods=["POST"]) 
def predict():
    area=float(request.values['area'])
    area=np.reshape(area,(-1,1))
    output=model.predict(area)
    output=output.item()
    output=round(output,2)
    
    return render_template("prediction.html",prediction_text=output)
if __name__=='__main__':
    app.run(port=8000)