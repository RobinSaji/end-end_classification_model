from operator import methodcaller
from optparse import Values
from pyexpat import model
from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))
app = Flask(__name__)


@app.route('/')
def home   ():
    return render_template('home.html')


@app.route('/predict',methods=["POST"])
def predict():

    float_features = [float(x) for x in  request.form.values()]
    features =[np.array(float_features)]
    classification = model.predict(features)
     
    return render_template('home.html',classification_text = "the flower type  is {} ".format(classification))



if __name__=="__main__":
    app.run(debug=True)