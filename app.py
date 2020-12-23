import numpy as np
from flask import Flask,request,jsonify,render_template
import  pickle
import os


# name of the application
app = Flask(__name__)


# load model from the pickel iris_model.pkl
def load_model():
	return pickle.load(open('iris_model.pkl','rb'))

@app.route('/')

# home page
def home():
	return render_template('index.html')


@app.route('/predict',methods=['POST'])	

def predict():

	labels=['setosa','versicolor','virginica']

	features=[float(x) for x in request.form.values()]

	values=[np.array(features)]

	model=load_model()

	prediction=model.predict(values)

	result=labels[prediction[0]]

	return render_template('index.html',output='The Flower is {}'.format(result))

if __name__ == '__main__':
    # app.debug = True
    app.run()


# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return 'Hello World!'

# if __name__ == '__main__':
#     app.debug = True
#     app.run()