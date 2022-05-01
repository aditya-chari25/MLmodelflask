from flask import Flask
from flask import render_template
from flask import request
import pandas as pd

app = Flask(__name__)

import pickle
model = pickle.load(open('rfcmodel.pkl', 'rb'))

name = 'hello world'
name1='about'

@app.route("/")
def base():
    return render_template('base.html', name=name)

@app.route("/hello")
def hello_world():
    return render_template('hello.html', name=name)

@app.route("/about",methods=['POST'])
def about():
    typem = request.form['type']
    timem = request.form['time']
    dof= request.form['dayofweek']
    mmodels = request.form['models']
    mparams = request.form['params']
    mqueuelen = request.form['queuelen']
    mtrails = request.form['trails']
    a = [[typem,timem,dof,mmodels,mparams,mqueuelen,mtrails]]
    a = pd.DataFrame(a)
    prediction =int(model.predict(a))
    return render_template('about.html', name=prediction)