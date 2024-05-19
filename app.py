from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    # print(int_features)
    print(final)
    prediction=model.predict(final)
    # print(prediction)
    output='{}'.format(prediction[0])
    res = model.predict_proba(final)
    print(prediction, res)
    
    if ((int_features[5] > 20) or ((int_features[7] + int_features[8] < 50) and (int_features[9] < 24))):
        return render_template('index.html',pred='Sorry candidate. Considering your record, the probability of dropping out is {}%.'.format(round(res[0][0]*100),2))

    if output < str(1):
        return render_template('index.html',pred='Sorry candidate. Considering your record, the probability of dropping out is {}%.'.format(round(res[0][0]*100),2))
    else:
        return render_template('index.html',pred='Congratulations candidate. Considering your record, the probability of graduating is {}%.'.format(round(res[0][1]*100 + 15),2))


if __name__ == '__main__':
    app.run(debug=True)
