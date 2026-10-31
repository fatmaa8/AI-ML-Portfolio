# NYC Taxi Fare Prediction

## Project Overview
This project aims to predict New York City taxi fares using machine learning regression models. The project uses a sample of 100,000 taxi trip records obtained from the NYC Open Data API.

## Dataset
- **Source:** [NYC Open Data API](https://data.cityofnewyork.us/resource/gkne-dk5s.json)
- **Initial size:** 100,000 rows and 18 columns
- **Target variable:** `fare_amount`

The dataset includes trip details such as pickup and dropoff locations, trip distance, passenger count, timestamps, and fare information.

## Project Workflow
1. Data collection using the NYC Open Data API.
2. Data cleaning and preprocessing.
3. Exploratory data analysis.
4. Feature engineering, including:
   - Pickup hour, day, and month
   - Day of the week
   - Weekend indicator
   - Geodesic distance between pickup and dropoff locations
   - Same-location indicator
5. Train-test splitting.
6. Model implementation and evaluation.

## Machine Learning Models

The project compares five regression models:

- Linear Regression — Scikit-learn
- Random Forest — Scikit-learn
- Linear Regression — From Scratch
- Decision Tree — From Scratch
- Random Forest — From Scratch

## Evaluation Metrics
The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

The final model comparison and results will be added after the final training runs.

## Technologies Used
- Python
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- Requests

## How to Run
1. Clone or download this repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook.
4. Run the cells in order.

## Project Goal
The main goal is to practice the complete machine learning workflow, compare different regression algorithms, and understand how machine learning models work through both library-based and from-scratch implementations.