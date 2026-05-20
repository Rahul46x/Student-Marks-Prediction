
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load('Students_mark_predictor_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():

    study_hours = float(request.form['study_hours'])

    # Validation
    if study_hours < 0 or study_hours > 12:
        return render_template(
            'index.html',
            prediction_text='Please enter valid study hours between 0 and 12'
        )

    # Convert input into numpy array
    features = np.array([[study_hours]])

    # Predict marks
    prediction = model.predict(features)

    # Convert numpy output to float
    prediction = round(float(prediction[0]), 2)



    # Limit prediction between 0 and 100
    prediction = max(0, min(prediction, 100))

    # Save data into CSV
    new_data = pd.DataFrame({
        'study_hours': [study_hours],
        'predicted_marks': [prediction]
    })

    new_data.to_csv(
        'smp_data_from_app.csv',
        mode='a',
        header=False,
        index=False
    )

    return render_template(
        'index.html',
        prediction_text=f'You will get approximately {prediction}% marks by studying {study_hours} hours per day.'
    )

if __name__ == "__main__":
    app.run(debug=True)

