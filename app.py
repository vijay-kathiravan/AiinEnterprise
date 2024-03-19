from flask import Flask, render_template, request
import ML_Classifier_model as mdl  # Import your trained machine learning model
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect inputs
    user_input1 = float(request.form["user_input1"])  # Convert string inputs to float
    user_input2 = float(request.form["user_input2"])
    user_input3 = float(request.form["user_input3"])
    user_input4 = float(request.form["user_input4"])
    user_input5 = float(request.form["user_input5"])
    user_input6 = float(request.form["user_input6"])

    # Create a DataFrame with the correct structure
    data = {
        'Weight': [user_input1],
        'Length1': [user_input2],
        'Length2': [user_input3],
        'Length3': [user_input4],
        'Height': [user_input5],
        'Width': [user_input6]
    }
    Data = pd.DataFrame(data)

    # Call model prediction function
    prediction = mdl.rf_classifier.predict(Data)

    # Convert numpy array to list for easy handling in template
    prediction_list = prediction.tolist()

    return render_template("result.html", prediction=prediction_list)

if __name__ == "__main__":
    app.run(debug=True)  # Remove debug=True before deployment
