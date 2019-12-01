import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
data=datasets.load_iris()

# Get data from the dataset object 'data'
dataArr=data.data

app = Flask(__name__)
model = pickle.load(open('human-life-balancing-indicator.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features = [int(x) for x in request.form.values()]	
	print("******************************")
	#print(int_features)
	unseen_x = np.array(int_features)	
	#print(unseen_x[0])

	
	import csv
	'''
	with open('writeToCsv.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["family", "emotional", "physical", "financial", "job"])
		writer.writerow([unseen_x[0], unseen_x[1], unseen_x[2], unseen_x[3], unseen_x[4]])
	'''
	
	# add row to CSV file
	with open('writeToCsv.csv', "a", newline='') as f:
		header = ["family", "emotional", "physical", "financial", "job"]
		writer = csv.DictWriter(f, fieldnames=header)
		writer.writerow({'family': unseen_x[0], 'emotional': unseen_x[1], 'physical': unseen_x[2], 'financial':unseen_x[3], 'job': unseen_x[4]})
	
	prob = model.predict([unseen_x])
	confidence_measure = model.predict_proba([unseen_x])[0, prob]
	#print("Unseen Data Accuracy on K-Fold: %.3f%%" % (confidence_measure*100.0))
	print(confidence_measure)
	pred_target = prob[0]
	
	if(pred_target == 1):
		text = "You are MORE HAPPIER person !!!"
	else:
		text = "You are LESS HAPPIER person !!! "
	
	return render_template('index.html', your_input = 'Your Input Scores: ----- ' + str(unseen_x), prob_accuracy = 'Model Confidence: ----- ' + str(confidence_measure),  prob_text= 'Model Result: ----- ' + str(text))	


if __name__ == "__main__":
    app.run(debug=True)