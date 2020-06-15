# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
model = pickle.load(open('model.pkl', 'rb'))
Tf_vect = pickle.load(open('Tfid.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = Tf_vect.transform(data).toarray()
    	my_prediction = model.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
    