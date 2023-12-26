import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
sc_model = pickle.load(open('models/scaler.pkl', 'rb'))
random_forest_model = pickle.load(open('models/random_forest.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # return request.form.values()
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_scaled = sc_model.transform(final_features)
    prediction = random_forest_model.predict(final_features)

    # output/= round(prediction[0], 2)
    if prediction == 1:
        answer = '"ML suggests potential autism traits in the child."'
    else :
        answer = '"ML analysis does not indicate traits associated with autism in the child."'
    return render_template('index.html', prediction_text = answer  )

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)

    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)