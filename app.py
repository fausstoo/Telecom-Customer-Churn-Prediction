from flask import Flask, request, render_template
import numpy as np
import pandas as pd


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
        return render_template('home.html')
    else:
        data = CustomData(
            CustomerID = int(request.form.get("CustomerID")),
            Age = int(request.form.get("Age")),
            Gender = request.form.get("Gender"), 
            Tenure = int(request.form.get("Tenure")),
            Usage_Frequency = int(request.form.get("Usage_Frequency")),
            Support_Calls = int(request.form.get("Support_Calls")),
            Payment_Delay = int(request.form.get("Payment_Delay")),
            Subscription_Type = request.form.get("Subscription_Type"),
            Contract_Length = request.form.get("Contract_Length"),
            Total_Spend = int(request.form.get("Total_Spend")),
            Last_Interaction = int(request.form.get("Last_Interaction"))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)