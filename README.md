# Car Price Prediction using Machine Learning

A machine learning regression project that predicts used-car prices from vehicle attributes such as model, year, mileage, transmission, fuel type, tax, MPG, and engine size.

## Project Overview

This project demonstrates an end-to-end introductory machine learning workflow:
- Dataset inspection and descriptive statistics
- Data-quality and missing-value checks
- Exploratory Data Analysis (EDA)
- Numerical and categorical feature analysis
- One-hot encoding of categorical variables
- Feature standardization
- Train-test splitting
- Linear Regression model training
- Model evaluation using R² and Adjusted R²

## Dataset

The supplied dataset contains **17,966 records and 9 columns**. The target variable is `price`.

| Feature | Type | Description |
|---|---|---|
| `model` | Categorical | Car model name |
| `year` | Numerical | Manufacturing/listing year |
| `price` | Numerical | Target car price |
| `transmission` | Categorical | Transmission type |
| `mileage` | Numerical | Vehicle mileage |
| `fuelType` | Categorical | Fuel type |
| `tax` | Numerical | Vehicle tax value |
| `mpg` | Numerical | Miles per gallon |
| `engineSize` | Numerical | Engine size |

No missing values were found in the supplied dataset.

## Exploratory Data Analysis

The notebook includes price distribution, correlation analysis, year vs. price, mileage vs. price, engine size vs. price, transmission vs. price, fuel type vs. price, and model vs. price visualizations.

## Machine Learning Workflow

### Feature Preparation

`price` is used as the target. The categorical columns `model`, `transmission`, and `fuelType` are converted using one-hot encoding with `drop_first=True`. The numerical columns `year`, `mileage`, `tax`, and `mpg` are standardized using `StandardScaler`.

### Model Training

The processed data is split into **67% training** and **33% testing** data with `random_state=42`. A scikit-learn **Linear Regression** model is trained on the training set.

## Results

| Metric | Score |
|---|---:|
| R² Score | **0.8402** |
| Adjusted R² Score | **0.8393** |

The model explains approximately 84% of the variation in the test-set target values according to R².

## Project Structure

```text
car-price-prediction-ml-project/
├── README.md
├── car_price_prediction.ipynb
├── data/
│   └── f.csv
├── requirements.txt
└── .gitignore
```

## Technologies

Python, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook.

## Installation

```bash
git clone https://github.com/anku04sh/car-price-prediction-ml-project.git
cd car-price-prediction-ml-project
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook car_price_prediction.ipynb
```

Run the notebook cells from top to bottom.

## Key Learning Outcomes

- Data analysis with pandas and NumPy
- Exploratory data analysis and visualization
- Categorical encoding and numerical scaling
- Regression model training and testing
- Model evaluation with R² and Adjusted R²




