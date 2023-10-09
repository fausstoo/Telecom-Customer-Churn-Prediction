import sys

from flask import Flask, request, render_template
import numpy as np
import pandas as pd


sys.path.append('../artifacts')

sys.path.append('../data')


from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Clear previous result when initially displaying the form
        results = None
        return render_template('home.html', results=results)
    else:
        data = CustomData(
            Age = float(request.form.get("Age")),
            Gender = request.form.get("Gender"), 
            Tenure = float(request.form.get("Tenure")),
            Usage_Frequency = float(request.form.get("Usage Frequency")),
            Support_Calls = float(request.form.get("Support Calls")),
            Payment_Delay = float(request.form.get("Payment Delay")),
            Subscription_Type = request.form.get("Subscription Type"),
            Contract_Length = request.form.get("Contract Length"),
            Total_Spend = float(request.form.get("Total Spend")),
            Last_Interaction = float(request.form.get("Last Interaction"))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)