from flask import Flask, render_template,request, jsonify
import pandas as pd
import pickle 
import numpy as np



app = Flask(__name__)

# Load the Iris Data Set
#df_1=pd.DataFrame(df)

	 
#if __name__ == "__main__":
 #   app.run(debug=True)

@app.route('/')

def home():
	return render_template("realtrial.html")



@app.route('/EDA.html/')
def EDA():
	#bar=create_plot
	return render_template("EDA.html")
	#print (df)
	#Age=df_1["age"].to_list()
	#bmi=df_1["BMI"].to_list()
	#legend = 'Monthly Data'
	
	#labels = ["January", "February", "March", "April", "May"]
	#values = [10, 9, 8, 7,4]
	#return render_template('EDA.html', values=bmi, labels=Age, legend=legend)
 
	#egend = 'Monthly Data'
	#labels = ["January", "February", "March", "April", "May"]
	#values = [10, 9, 8, 7,4]
	#return render_template('eda.html', values=values, labels=labels, legend=legend)
	
	#return render_template("EDA.html")


@app.route('/visualization.html/')
def visualization():
	return render_template("visualization.html") 

@app.route('/result.html/')
def result():
	return render_template("result.html")    

@app.route('/predict.html/')
def predict():
	return render_template("predict.html")

model=pickle.load(open("model.pkl",'rb'))
@app.route('/prediction', methods=['POST','GET'])
def prediction():
	features=["gen","age","cgspd","ps","ph","db","totChol","sysBP","BMI","gc"]
	int_features=[]
	for i in features:
		int_features.append(request.form.get(i))
	int_features = list(map(int,int_features))
	prediction = model.predict([int_features])
	output=str(prediction[0])
	#return render_template("predict.html" ,prediction_text='{}'.format(output))

	if int(output)==1:			
		return render_template("predict.html",prediction_text='There is a risk of heart disease')
	else:	
		return render_template("predict.html",prediction_text='There is no risk of heart disease')

		
#smodel = pickle.load(open('model_fitting.pkl','rb'))
#def predict():
	#int_features = [int(x) for x in request.form.values()]
	#final_features = [np.array(int_features)]
	#prediction = model.predict(final_features)
	#output = prediction[0]
	#return render_template("prediction.html" ,prediction_text='{}'.format(output))

#df=pd.read_csv("heart disease.csv",names=["BMI","totChol","TenYearCHD"])
#feature_names=df.columns[0:-1].values.tolist()
#create plot


if __name__ == '__main__':
	app.run(debug=True) 