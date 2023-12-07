import pickle
from flask import Flask ,request ,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = pickle.load(open('models/linear.pkl','rb'))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    
    
    if request.method ==  'POST':
        temp = int(request.form.get('temp'))
        Rh = int(request.form.get('Rh'))
        Ws = int(request.form.get('Ws'))
        Rain = int(request.form.get('Rain'))

        FFMC = int(request.form.get('FFMC'))
        DMC = int(request.form.get('DMC'))
        ISI = int(request.form.get('ISI'))
        Class = int(request.form.get('Class'))
        Region = int(request.form.get('Region'))

        

        
        result = model.predict([[temp,Rh,Ws,Rain,FFMC,DMC,ISI,Class,Region]])
        return render_template('home.html',   prediction_text='The prediction of forest fire is {}'.format(result))

    else:
        return render_template('index.html')
        
    
if __name__ == '__main__':
    app.run(host="0.0.0.0")