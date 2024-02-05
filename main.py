# Import Libraries and Modules
import pickle
import pandas as pd
from fastapi import FastAPI

# Instantiate app
app = FastAPI()

# Load Model Pipeline
app.state.model = pickle.load(open('denseNN_231214.pkl', 'rb'))

# Get Economic Data into DF
url = 'https://raw.githubusercontent.com/hgobinds/HDB_data/9a824522e6112704fd902c933d90d9691e70cf3c/sg_econ_data_historical_future.csv'
X_future_econ = pd.read_csv(url)

X_pred_cols = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model',
    'lease_commence_date', 'sold_year', 'sold_remaining_lease',
    'max_floor_lvl', '5 year bond yields', 'GDPm (Current Prices)',
    'GDP per capita', 'Personal Income m', 'Unemployment Rate',
    'Core inflation', 'Median Household Inc',
    'Lime, Cement, & Fabricated Construction Materials Excl Glass & Clay Materials',
    'Clay Construction Materials & Refractory Construction Materials',
    'most_closest_mrt', 'walking_time_mrt', 'ResidentPopulation',
    'ResidentPopulation_Growth_Rate']

# Predict Function
@app.get("/predict")
def predict(year: int,
            town: str,
            flat_type: str,
            storey_range: str,
            floor_area_sqm: str,
            flat_model: str,
            lease_commence_date: int,
            sold_remaining_lease: int,
            max_floor_lvl: int,
            most_closest_mrt: str,
            walking_time_mrt: int
            ):
    """
    Makes a single prediction for HDB resale price
    """

    # Reshape Economic data to year required
    X_future_econ_year = X_future_econ[X_future_econ['year'] == int(year)]

    # Create prediction dataframe
    X_hdb_pred = pd.DataFrame({"town" :[town],
                               "flat_type":[flat_type],
                               "storey_range":[storey_range],
                               "floor_area_sqm":[floor_area_sqm],
                               "flat_model":[flat_model],
                               "lease_commence_date":[lease_commence_date],
                               "sold_year" : [year],
                               "sold_remaining_lease" : [sold_remaining_lease],
                               "max_floor_lvl" : [max_floor_lvl],
                               "most_closest_mrt" : [most_closest_mrt],
                               "walking_time_mrt": [walking_time_mrt]
                              })

    # Join Econ data
    X_predict = X_hdb_pred.merge(X_future_econ_year, how='left', left_on='sold_year', right_on='year')

    # Reorientate columns
    X_predict = X_predict[X_pred_cols]

    # Get prediction from model pipeline
    result = round(float(app.state.model.predict(X_predict)))

    # Return as Dict
    return {'hdb_pricing': result}

    # To test predict function run the following url:
    # http://127.0.0.1:8000/predict?year=2028&town=HOUGANG&flat_type=3 ROOM&storey_range=13 TO 15&floor_area_sqm=95&flat_model=Simplified&lease_commence_date=1980&sold_remaining_lease=93&max_floor_lvl=12&most_closest_mrt=KALLANG&walking_time_mrt=1500

# Predict Range
@app.get("/fullpredict")
def predict(year: int,
            town: str,
            flat_type: str,
            storey_range: str,
            floor_area_sqm: str,
            flat_model: str,
            lease_commence_date: int,
            sold_remaining_lease: int,
            max_floor_lvl: int,
            most_closest_mrt: str,
            walking_time_mrt: int
            ):
    """
    Makes a range of predictions for HDB resale price from year the HDB lease started - 2033
    """
    # Reshape Economic data to year required
    # + 4 cause HDBs can be sold 5 years after purchase. (4 + year bought)
    X_future_econ_year = X_future_econ[X_future_econ['year'] >= int(lease_commence_date) + 4]

    # Create prediction dataframe
    X_hdb_pred = pd.DataFrame({ "town" :[town],
                                "flat_type":[flat_type],
                                "storey_range":[storey_range],
                                "floor_area_sqm":[floor_area_sqm],
                                "flat_model":[flat_model],
                                "lease_commence_date":[lease_commence_date],
                                "max_floor_lvl" : [max_floor_lvl],
                                "most_closest_mrt" : [most_closest_mrt],
                                "walking_time_mrt": [walking_time_mrt]
                                })

    # Create variables to use in the creating of the prediction data frame
    resale_year = X_hdb_pred['lease_commence_date'][0] + 4
    factor = 2034 - X_hdb_pred['lease_commence_date'][0] - 4

    # Multiply single row in df to include the year range
    X_hdb_pred = pd.concat([X_hdb_pred] * factor, ignore_index=True)

    # Create year column using factor created earlier
    year_range = range(resale_year, resale_year + factor)
    X_hdb_pred['sold_year'] = year_range

    # Create remaining lease column for prediction years
    X_hdb_pred['sold_remaining_lease'] = 99 - (X_hdb_pred['sold_year'] - X_hdb_pred['lease_commence_date'])

    # Remove any dates before 1990 as it creates NaNs
    X_hdb_pred = X_hdb_pred[X_hdb_pred['sold_year'] >= 1990].reset_index(drop=True)

    # Join Econ data
    X_predict = X_hdb_pred.merge(X_future_econ_year, how='right', left_on='sold_year', right_on='year')

    # Reorientate columns
    X_predict = X_predict[X_pred_cols]

    # Get using model pipeline
    X_predict['forecast'] = app.state.model.predict(X_predict)

    # Round the results
    X_predict['forecast'] = X_predict['forecast'].round()

    # Extract Results
    result = X_predict[['sold_year', 'forecast']]

    # Return as Dict
    return result.to_dict(orient='index')

    # to test, run url below
    # http://127.0.0.1:8000/fullpredict?year=2028&town=HOUGANG&flat_type=3%20ROOM&storey_range=13%20TO%2015&floor_area_sqm=95&flat_model=Simplified&lease_commence_date=1980&sold_remaining_lease=93&max_floor_lvl=12&most_closest_mrt=KALLANG&walking_time_mrt=1500
