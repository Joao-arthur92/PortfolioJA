from flask import Flask, request
import pickle
import pandas as pd

from wine_quality import WineQuality

# load model
model = pickle.load(open('C:\\Users\\ja_me\\Projetos_Portfolio\\model_wine_quality.pkl', 'rb'))

# Instanciante flask
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    test_json = request.get_json()

    # collect data
    if test_json:
        if isinstance(test_json, dict):  # unique value
            df_raw = pd.DataFrame(test_json, index=[0])
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

    # Instanciate data preparation
    pipeline = WineQuality()

    # data preparation
    df1 = pipeline.data_preparation(df_raw)

    # prediction
    pred = model.predict(df1)

    # response
    df1['prediction'] = pred
    return df1.to_json(orient='records')


if __name__ == '__main__':
    # start flask
    app.run(host='192.168.0.112', port='5000')
