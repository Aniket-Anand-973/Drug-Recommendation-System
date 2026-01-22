from flask import Flask,render_template,url_for,request

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 


import pickle

app = Flask(__name__)

random = pickle.load(open('random.pkl','rb'))

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset)
        return render_template("preview.html",df_view = df)


@app.route('/prediction')
def prediction():
    return render_template("prediction.html")

@app.route('/predict',methods=["POST"])
def predict():
    if request.method == 'POST':
        Itching = request.form['Itching']
        skin_rash = request.form['skin_rash']
        nodal_skin_eruptions = request.form['nodal_skin_eruptions']
        continuous_sneezing = request.form['continuous_sneezing']
        shivering = request.form['shivering'] 
        chills = request.form['chills']
        stomach_pain = request.form['stomach_pain']
        ulcers_on_tongue = request.form['ulcers_on_tongue']
        vomiting = request.form['vomiting']
        cough = request.form['cough']
        chest_pain = request.form['chest_pain']

        yellowish_skin = request.form['yellowish_skin']
        loss_of_appetite = request.form['loss_of_appetite']
        abdominal_pain = request.form['abdominal_pain']
        yellow_urine = request.form['yellow_urine']
        weight_loss = request.form['weight_loss'] 
        restlessness = request.form['restlessness']
        irregular_sugar_level = request.form['irregular_sugar_level']
        excessive_hunger = request.form['excessive_hunger']
        increased_appetite = request.form['increased_appetite']
        high_fever = request.form['high_fever']
        headache = request.form['headache']


        diarrhoea = request.form['diarrhoea']
        muscle_pain = request.form['muscle_pain']
        red_spots_over_body = request.form['red_spots_over_body'] 
        runny_nose = request.form['runny_nose']
        breathlessness = request.form['breathlessness']
        fast_heart_rate = request.form['fast_heart_rate']
        dark_urine = request.form['dark_urine']
        
        
        
        model = request.form['model']
        
		# Clean the data by convert from unicode to float 
        
        sample_data = [Itching,skin_rash,nodal_skin_eruptions,continuous_sneezing,shivering,chills,stomach_pain,ulcers_on_tongue,vomiting,cough,chest_pain,
        yellowish_skin,loss_of_appetite,abdominal_pain,yellow_urine,weight_loss,restlessness,irregular_sugar_level,excessive_hunger,increased_appetite,high_fever,headache,
        diarrhoea,muscle_pain,red_spots_over_body,runny_nose,breathlessness,fast_heart_rate,dark_urine]
        # clean_data = [float(i) for i in sample_data]
        # int_feature = [x for x in sample_data]
        int_feature = [float(i) for i in sample_data]
        print(int_feature)
    

		# Reshape the Data as a Sample not Individual Features
        
        ex1 = np.array(int_feature).reshape(1,-1)
        print(ex1)
		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

        # Reloading the Model
        if model == 'RandomForestClassifier':
           result_prediction = random.predict(ex1)
           
            
        elif model == 'DecisionTreeClassifier':
          result_prediction = decision.predict(ex1)
           
           
        
        # if result_prediction ==1:
        #     result = 'Fake'
        # else:
        #     result = 'Real'    

          

    return render_template('prediction.html', prediction_text= result_prediction[0], model = model)

@app.route('/performance')
def performance():
    return render_template("performance.html")

@app.route('/chart')
def chart():
    return render_template("chart.html")    

if __name__ == '__main__':
	app.run(debug=True)
