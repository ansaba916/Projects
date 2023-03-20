# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:42:00 2023

@author: 91974
"""

from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route("/")
def home():
    return render_template("rent.html")
@app.route("/predict", methods=['POST'])
def predict():
    
    BHK=(request.values["text1"])
    #print(BHK)
    size=(request.values["text2"])
    #print(size)
    
    bedroom=(request.values["text3"])
    #print(bedroom)
    
    a=pd.DataFrame({"BHK":[BHK],
                "Size":[size],
                "Bathroom":[bedroom]})
    print(a)
    y_pred=(model.predict(a))
    print(y_pred)

    
    
    
    
    
    return render_template('rent.html',prediction_text="rent is ${}".format(y_pred))
if __name__=='__main__':
    app.run(port=8000)