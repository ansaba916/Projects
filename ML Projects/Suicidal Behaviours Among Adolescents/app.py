# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:01:16 2023

@author: 91974
"""

from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('model.plk','rb'))
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/upload1")
def home1():
    return render_template("index.html")
@app.route("/upload2")
def home2():
    return render_template("signup.html")
@app.route("/upload3")
def home3():
    return render_template("form.html")

@app.route("/predict",methods=['POST'])
def predict():
    Year=(request.values['text1'])
    AgeGroup=(request.values['text2'])
    Sex=(request.values['text3'])
    Currently_Drink_Alcohol=(request.values['text4'])
    Really_Get_Drunk=(request.values['text5'])
    Overweight=(request.values['text6'])
    Use_Marijuana=(request.values['text7'])
    Have_understanding_Parents=(request.values['text8'])
    Missed_classes_without_permission=(request.values['text9'])
    Had_sexual_relation=(request.values['text10'])
    Smoke_cig_currently=(request.values['text11'])
    Had_fights=(request.values['text12'])
    Bullied=(request.values['text12'])
    Got_seriously_injured=(request.values['text14'])
    No_close_friends=(request.values['text15'])
    x=pd.DataFrame({"Year":[Year],
                    "AgeGroup":[AgeGroup],
                    "Sex":[Sex],
                    "Currently_Drink_Alcohol":[Currently_Drink_Alcohol],
                    "Really_Get_Drunk":[Really_Get_Drunk],
                    "Overweight":[Overweight],
                    "Use_Marijuana":[Use_Marijuana],
                    "Have_understanding_Parents":[Have_understanding_Parents],
                    "Missed_classes_without_permission":[Missed_classes_without_permission],
                    "Had_sexual_relation":[Had_sexual_relation],
                    "Smoke_cig_currently":[Smoke_cig_currently],
                    "Had_fights":[Had_fights],
                    "Bullied":[Bullied],
                    "Got_seriously_injured":[Got_seriously_injured],
                    " No_close_friends":[No_close_friends]})

    print(x)
    y_pred=model.predict(x)
    print(y_pred)
    return render_template("result.html", prediction_text="The prediction is {}".format(y_pred))
if __name__=="__main__":
        app.run()
