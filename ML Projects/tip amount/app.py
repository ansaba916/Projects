
from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route("/")
def home():
    return render_template("tip.html")
@app.route("/predict", methods=['POST'])
def predict():
    price=float(request.values["text"])
    
    price=np.reshape(price,(-1,1))
    output=model.predict(price)
    output=output.item()
    output=round(output,2)
    
    return render_template('tip.html',prediction_text="price for the house is ${}".format(output))
if __name__=='__main__':
    app.run(port=8000)