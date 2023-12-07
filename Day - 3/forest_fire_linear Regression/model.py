import pickle
import numpy as np
import pandas as pd
ridge = pickle.load(open('models/ridge.pkl','rb'))
standard_model = pickle.load(open('models/scaler2.pkl','rb'))
new_data_scaled = standard_model.transform([[1,1,1,1,1,1,1,1,1]])
result = ridge.predict(new_data_scaled) 
print(result)