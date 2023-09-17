import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
application = Flask(__name__)
app=application

filename = 'model.sav'
load_model = pickle.load(open(filename, 'rb'))

@app.route("/")  
def index():  
    return render_template('index.html')  #render_temlate look for a file(index.html in this case) in templete folder

@app.route('/predictdata',methods=['GET','POST'])  
 
def predict_datapoint():
    if request.method=='POST':
        Age=float(request.form.get('Age'))
        Subscription_Length_Months = float(request.form.get('Subscription_Length_Months'))
        Monthly_Bill = float(request.form.get('Monthly_Bill'))
        Total_Usage_GB = float(request.form.get('Total_Usage_GB'))
         # Create a numpy array with the input data
        input_data = np.array([[Age, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB]])

        # Use the StandardScaler to transform the input data
        scaled_data = scaler.fit_transform(input_data)

        # Make predictions using the loaded model
        result = load_model.predict(scaled_data)
        print("Result =", result)

        return render_template('home.html', result=result[0])

       # data=StandardScaler.fit_transform([[Age, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB]])

        #result=load_model.predict(data)
        #print("Result=",result)

        #return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)