from flask import Flask, render_template,request, jsonify
import pandas as pd
import pickle 
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
	return render_template("realtrial.html")


@app.route('/EDA.html/')
def EDA():
	return render_template("EDA.html")   

@app.route('/predict.html/')
def predict():
	return render_template("predict.html")

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    if request.method=='POST':
        result=request.form

		#Prepare the feature vector for prediction
        pkl_file = open('cat', 'rb')
        index_dict = pickle.load(pkl_file)
        new_vector = np.zeros(len(index_dict))

        try:
        	new_vector[index_dict['age'+str(result['age'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['capital_gain'+str(result['cg'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['capital_loss'+str(result['cl'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['hours_per_week'+str(result['hrs'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['work_class'+str(result['wc'])]] = 1
        except:
            pass
        try:
        	new_vector[index_dict['education'+str(result['edu'])]] = 1
        except:
        	pass
        try:
        	new_vector[index_dict['marital_status'+str(result['marital_status'])]] = 1
        except:
        	pass
        try:
        	new_vector[index_dict['native_country'+str(result['native'])]] = 1
        except:
        	pass
        try:
        	new_vector[index_dict['occupation'+str(result['occupation'])]] = 1
        except:
        	pass
        try:
        	new_vector[index_dict['age'+str(result['age'])]] = 1
        except:
        	pass
        try:
        	new_vector[index_dict['race'+str(result['race'])]] = 1
        except:
        	pass
        try:
        	new_vector[index_dict['relationship'+str(result['relationship'])]] = 1
        except:
        	pass
        
        new_vector = new_vector.reshape(-1,1)
        pkl_file = open('logreg.pkl', 'rb')
        logreg = pickle.load(pkl_file)
        prediction = logreg.predict(new_vector)
        
        return render_template('result.html',prediction=prediction)

	#features=["age","wc","edu","Pvt","Govt","occu","race","gen","cg","cl","hrs","native"]
	#int_features=[]
	#for i in features:
	#	int_features.append(request.form.get(i))
	#int_features = list(map(int,int_features))
	#int_features=[int(x) for x in request.form.values()]
	#prediction = model.predict([int_features])
	#output=str(prediction[0])
	#if prediction==1:
	#	output="Income is greater than 50K"
	#else:
	#	output="Income is less than 50K"

	#return render_template("predict.html" ,prediction_text='{}'.format(output))

if __name__ == '__main__':
	app.run(debug=True) 