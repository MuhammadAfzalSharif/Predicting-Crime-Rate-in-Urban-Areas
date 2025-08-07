# Import required libraries
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask application
application = Flask(__name__)
app = application

# Load the trained Elastic Net model and StandardScaler
try:
    ElasticNet_CV = pickle.load(open('models/Elastic_Net_CV.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    raise

# Define route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for prediction with GET and POST methods
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Retrieve input features from the form
            population_density = float(request.form.get('Population Density'))
            unemployment_rate = float(request.form.get('Unemployment Rate %'))
            police_station_distance = float(request.form.get('Police Station Distance km'))
            number_of_schools = float(request.form.get('Number of Schools'))
            average_income = float(request.form.get('Average Income'))
            age_group_15_24 = float(request.form.get('Age Group Ratio (15-24) %'))
            age_group_24_44 = float(request.form.get('Age Group Ratio (24-44) %'))
            age_group_44_plus = float(request.form.get('Age Group Ratio (44+) %'))
            poverty_rate = float(request.form.get('Poverty Rate %'))
            school_dropout_rate = float(request.form.get('School Dropout Rate %'))
            good_housing_quality = float(request.form.get('Good Housing Quality %'))
            distance_to_public_transport = float(request.form.get('Distance to Public Transport km'))
            recreational_facilities = float(request.form.get('Recreational Facilities'))
            child_single_parent_rate = float(request.form.get('Child Living With Single Parent Rate %'))
            cctv_cameras = float(request.form.get('CCTV Cameras'))
            drug_hospital_admissions = float(request.form.get('Drug Hospital Admissions'))
            police_patrol_frequency = float(request.form.get('Police Patrol Frequency'))
            emergency_response_time = float(request.form.get('Emergency Response Time (Mins)'))
            youth_unemployment_rate = float(request.form.get('Youth Unemployment Rate %'))
            alcohol_outlet_density = float(request.form.get('Alcohol OutletDensity'))
            distance_to_high_risk_areas = float(request.form.get('Distance to High Risk_Areas km'))
            mental_health_services_count = float(request.form.get('Mental Health Services Count'))

            # Create a pandas DataFrame with feature names to match training
            feature_names = [
                'Population Density', 'Unemployment Rate %', 'Police Station Distance km',
                'Number of Schools', 'Average Income', 'Age Group Ratio (15-24) %',
                'Age Group Ratio (24-44) %', 'Age Group Ratio (44+) %', 'Poverty Rate %',
                'School Dropout Rate %', 'Good Housing Quality %', 'Distance to Public Transport km',
                'Recreational Facilities', 'Child Living With Single Parent Rate %',
                'CCTV Cameras', 'Drug Hospital Admissions', 'Police Patrol Frequency',
                'Emergency Response Time (Mins)', 'Youth Unemployment Rate %',
                'Alcohol OutletDensity', 'Distance to High Risk_Areas km',
                'Mental Health Services Count'
            ]
            input_data = pd.DataFrame([[
                population_density, unemployment_rate, police_station_distance,
                number_of_schools, average_income, age_group_15_24,
                age_group_24_44, age_group_44_plus, poverty_rate,
                school_dropout_rate, good_housing_quality, distance_to_public_transport,
                recreational_facilities, child_single_parent_rate, cctv_cameras,
                drug_hospital_admissions, police_patrol_frequency, emergency_response_time,
                youth_unemployment_rate, alcohol_outlet_density, distance_to_high_risk_areas,
                mental_health_services_count
            ]], columns=feature_names)

            # Scale the input data using the loaded StandardScaler
            scaled_data = standard_scaler.transform(input_data)

            # Predict using the Elastic Net model
            prediction = ElasticNet_CV.predict(scaled_data)

            # Extract the first value from the prediction
            result = prediction[0].astype(int)

            # Render the home.html template with the prediction result
            return render_template('home.html', results=result)
        except ValueError:
            # Handle invalid input (e.g., non-numeric values)
            return render_template('home.html', results=None, error="Please enter valid numeric values for all fields.")
    else:
        # For GET request, render the form page without any prediction
        return render_template('home.html', results=None)

# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)