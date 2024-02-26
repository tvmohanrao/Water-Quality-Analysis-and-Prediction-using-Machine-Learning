from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
# Load the trained model and scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('min_max_scaler.joblib')

# Define the prediction function
def predict_potability(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity):
    # Load PolynomialFeatures
    Polynom = PolynomialFeatures(degree=3)

    # Transform the input using the same scaler and polynomial features
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    input_transformed = scaler.transform(Polynom.fit_transform(input_data))

    # Make a prediction
    prediction = model.predict(input_transformed)

    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        ph = float(request.form['ph'])
        hardness = float(request.form['hardness'])
        solids = float(request.form['solids'])
        chloramines = float(request.form['chloramines'])
        sulfate = float(request.form['sulfate'])
        conductivity = float(request.form['conductivity'])
        organic_carbon = float(request.form['organic_carbon'])
        trihalomethanes = float(request.form['trihalomethanes'])
        turbidity = float(request.form['turbidity'])

        # Get prediction
        prediction = predict_potability(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity)

        return render_template('index.html', prediction=prediction, ph=ph, hardness=hardness, solids=solids, chloramines=chloramines,
                               sulfate=sulfate, conductivity=conductivity, organic_carbon=organic_carbon,
                               trihalomethanes=trihalomethanes, turbidity=turbidity)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
