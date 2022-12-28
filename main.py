from flask import Flask, render_template ,request
import pandas as pd
import pickle
from joblib import dump, load

app = Flask(__name__)
car = pd.read_csv('Cleaned_cars.csv')
model = pickle.load(open("XGBOOST.pkl", "rb"))

@app.route('/')
def index():
    companies = sorted(car['Company'].unique())
    cars_models = sorted(car['Model_name'].unique())
    year = sorted(car['Model_year'].unique())
    fuel_types = sorted(car['Fuel_Type'].unique())
    transmissions = sorted(car['Transmission'].unique())
    owners = sorted(car['Owner'].unique())
    registration = sorted(car['Registration'].unique())
    state = sorted(car['State'].unique())
    seating_capacity = sorted(car['Seating_capacity'].unique())
    city = sorted(car['City'].unique())
    companies.insert(0," ")
    cars_models.insert(0, " ")
    year.insert(0, " ")
    fuel_types.insert(0, " ")
    transmissions.insert(0, " ")
    owners.insert(0, " ")
    registration.insert(0, " ")
    state.insert(0, " ")
    seating_capacity.insert(0, " ")
    city.insert(0, " ")


    return render_template('index.html', company=companies, car_model=cars_models, years=year, fuel_Type=fuel_types, transmission =transmissions, owner=owners , registration=registration ,state=state ,seating_capacity=seating_capacity,city=city)


@app.route('/predict', methods=['POST'])
def predict():
    Company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    kms_driven = int(request.form.get('kilo_driven'))
    owner = request.form.get('owner')
    transmission = request.form.get('transmission')
    registration = request.form.get('registration')
    state = request.form.get('state')
    fuel_type = request.form.get('fuel_Type')
    mileage = float(request.form.get('mileage'))
    fuel_capacity = int(request.form.get('fuel_capacity'))
    seating_capacity = int(request.form.get('seating_capacity'))
    city = request.form.get('city')



    prediction = model.predict(pd.DataFrame([[Company, car_model ,year, kms_driven ,owner , transmission,registration , state,fuel_type,mileage,fuel_capacity,seating_capacity ,city]], columns=['Company', 'Model_name','Model_year','Kilometers','Owner','Transmission','Registration','State','Fuel_Type','Mileage','Fuel_capacity','Seating_capacity','City']))

    return str(prediction[0])



if __name__ == '__main__':
    app.run(debug=True)

