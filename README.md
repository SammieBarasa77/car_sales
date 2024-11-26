# Analtyics & Machine Learning Project: Car Sales analysis and Prediction
![cover](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/cover_final.png)

# Table of Contents

1. [Introduction](#introduction)
2. [Data Import](#data-import)
3. [Data Understanding](#data-understanding)
4. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
   - [Handle Missing Data](#handle-missing-data)
   - [Check for Missing Values](#check-for-missing-values)
   - [Filtering Outliers](#filtering-outliers)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Monthly Sales Revenue Trends](#monthly-sales-revenue-trends)
   - [Top Car Models by Price](#top-car-models-by-price)
   - [Regional Performance (Sales by Region)](#regional-performance-sales-by-region)
6. [Feature Engineering](#feature-engineering)
   - [Income-to-Price Ratio](#income-to-price-ratio)
   - [Price Category Based on Price Ranges](#price-category-based-on-price-ranges)
7. [Predictive Analytics](#predictive-analytics)
   - [Using Machine Learning to Predict Sales](#using-machine-learning-to-predict-sales)
   - [Visualizing the Prediction](#visualizing-the-prediction)
8. [Key Metrics for Evaluation](#key-metrics-for-evaluation)
   - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
   - [Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)](#mean-squared-error-mse-root-mean-squared-error-rmse)
   - [R² Score](#r²-score)
   - [Train-Test Performance Comparison](#train-test-performance-comparison)
   - [Feature Importance Analysis](#feature-importance-analysis)
   - [Cross-validation](#cross-validation)
   - [Residual Analysis](#residual-analysis)
9. [Other Analyses](#other-analyses)
   - [Price Distribution by Car Model Analysis](#price-distribution-by-car-model-analysis)
   - [Seasonality of Sales](#seasonality-of-sales)
   - [Price Sensitivity (Price vs Sales Volume)](#price-sensitivity-price-vs-sales-volume)
   - [Car Model Popularity (Most Sold Models)](#car-model-popularity-most-sold-models)
   - [Customer Segmentation (Clustering)](#customer-segmentation-clustering)
10. [Insights, Findings, and Recommendations](#insights-findings-and-recommendations)

## Introduction

## Data Import
```python
car_data = pd.read_csv(r"C:\Users\samue\Downloads\Car Sales.xlsx - car_data.csv")  
car_data
```

## Data Understanding

Exploring the dataset for basic statistics and data types and checking for unique values in crucial columns like Company, Model, and Dealer_Region.
```python
# Overview of the dataset
print(car_data.describe(include='all'))
print("\nUnique values per column:")

for column in ['Company', 'Model', 'Dealer_Region']:
    print(f"{column}: {car_data[column].nunique()} unique values")
```
## Data Cleaning and Preprocessing

### Handle Missing Data

### Checking for Missing Values

```python
missing_values = car_data.isnull().sum()
print("Missing values:\n", missing_values)
```
### Filtering outliers out
```python
# Recalculate the IQR, Q1, Q3, and boundaries
Q1 = car_data['Price ($)'].quantile(0.25)
Q3 = car_data['Price ($)'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove all outliers beyond the bounds
filtered_data = car_data[(car_data['Price ($)'] >= lower_bound) & (car_data['Price ($)'] <= upper_bound)]

# Confirm no data points exceed the bounds
print(f"New dataset size after strict filtering: {filtered_data.shape}")
print(f"Any remaining outliers above upper bound? {filtered_data['Price ($)'].max() > upper_bound}")
print(f"Any remaining outliers below lower bound? {filtered_data['Price ($)'].min() < lower_bound}")
```

Box pplot without outliers
```python
sns.boxplot(x=car_data_filtered['Price ($)'])
plt.title("Boxplot for Car Prices (Without Outliers)")
plt.show()
```

```python
# Updating the dataset
car_data = filtered_data
```

Visualizations to identify price and income outliers.
```python

import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot for Price
sns.boxplot(x=car_data['Price ($)'])
plt.title("Boxplot for Car Prices")
plt.show()
```

## Exploratory Data Analysis (EDA)
### Monthly sales Revenue Trends 
```python
# Trends Over Time: Sales volume and revenue over time.

# Convert the Date column to datetime format
car_data['Date'] = pd.to_datetime(car_data['Date'], errors='coerce')

# Drop rows with invalid dates (if any)
car_data = car_data.dropna(subset=['Date'])

#Group by month and calculate monthly sales revenue
sales_overtime = car_data.groupby(car_data['Date'].dt.to_period('M'))['Price ($)'].sum()

# Plot the results
sales_overtime.plot(kind='line', title="Monthly Sales Revenue", ylabel="Revenue ($)", xlabel="Month")
plt.xticks(rotation=45)
plt.show()
```
### Top Car Models by Price
```python
# Aggregate data to get total price for each car model
top_models_by_price = car_data.groupby('Model')['Price ($)'].sum().sort_values(ascending=False).head(10)

# Plot the top models by price
top_models_by_price.plot(kind='barh', title="Top Car Models by Price", xlabel="Total Price ($)", ylabel="Car Model", color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()
```

### Regional Performance (Sales by Region)
```python
# Sales by region
region_sales = car_data.groupby('Dealer_Region')['Price ($)'].sum().sort_values(ascending=False)
region_sales.plot(kind='bar', title="Sales by Region", ylabel="Revenue ($)")
plt.show()
```

## Feature Engineering
### Income-to-Price Ratio
```python

# Add Price Category
bins = [0, 20000, 40000, 60000, float('inf')]
labels = ['Budget', 'Midrange', 'Premium', 'Luxury']
car_data['Price_Category'] = pd.cut(car_data['Price ($)'], bins=bins, labels=labels)

# Add Income-to-Price Ratio
car_data['Income_to_Price_Ratio'] = car_data['Annual Income'] / car_data['Price ($)']

car_data
```

## Predictive Analytics

### Using Machine Learning to Predict Sales
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Features and target
features = car_data[['Annual Income', 'Price ($)']]
target = car_data['Price ($)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Training the  model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predicting
predictions = model.predict(X_test)
```

#### Visualizing the Prediction
```python
# Create a DataFrame for comparison
comparison_df = pd.DataFrame({
    'Actual Sales': y_test,
    'Predicted Sales': predictions
}, index=y_test.index)

# Sort by index for a clear trend (optional)
comparison_df = comparison_df.sort_index()

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(comparison_df['Actual Sales'], label="Actual Sales", color='blue', marker='o')
plt.plot(comparison_df['Predicted Sales'], label="Predicted Sales", color='red', linestyle='--', marker='x')
plt.title("Actual vs Predicted Sales")
plt.xlabel("Index")
plt.ylabel("Sales ($)")
plt.legend()
plt.grid(True)
plt.show()
```

## Key Metrics for Evaluation

### Mean Absolute Error (MAE)
```python
# Measures the average absolute difference between actual and predicted values. Lower is better.

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae}")
```

### Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
```

### R2 Score
```python

from sklearn.metrics import r2_score

r2 = r2_score(y_test, predictions)
print(f"R-squared (R²): {r2}")
```

### Train - Test Performance  Comparison

```python

r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)

print(f"R-squared (Train): {r2_train}")
print(f"R-squared (Test): {r2_test}")
```

### Feature Importance Analysis
```python
import pandas as pd

importance = model.feature_importances_
features_list = features.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': features_list, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print or visualize the feature importances
print(importance_df)

# Visualization
import matplotlib.pyplot as plt
importance_df.plot(kind='barh', x='Feature', y='Importance', legend=False)
plt.title('Feature Importance')
plt.show()
```

### Cross-validation
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, features, target, cv=5, scoring='r2')

print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean()}")
```

### Residual Analysis
```python
import seaborn as sns
import numpy as np

residuals = y_test - predictions

# Histogram of residuals
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.show()
```

```python
# Scatter plot of residuals
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals vs Actuals")
plt.xlabel("Actual Sales")
plt.ylabel("Residuals")
plt.show()
```

## Other Analyses
### Price Distribution by Car Model Analysis
```python
# Top 10 models 

# Get the top 10 car models based on the count of sales
top_10_models = car_data['Model'].value_counts().head(10).index

# Filter the dataset to include only the top 10 models
top_10_car_data = car_data[car_data['Model'].isin(top_10_models)]

# Price distribution by top 10 car models
plt.figure(figsize=(12, 8))
sns.boxplot(x='Model', y='Price ($)', data=top_10_car_data)
plt.xticks(rotation=90)
plt.title("Price Distribution by Top 10 Car Models")
plt.xlabel("Car Model")
plt.ylabel("Price ($)")
plt.show()
```

### Seasonality of sales
```python
# Extract month from the 'Date' column 
car_data['Month'] = car_data['Date'].dt.month

# Sales by month
monthly_sales = car_data.groupby('Month')['Price ($)'].sum()

# Visualization
monthly_sales.plot(kind='line', title="Monthly Sales Trends", ylabel="Total Sales ($)", figsize=(10, 6))
plt.xticks(monthly_sales.index, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.show()
```

### Price Sensitivity(Price vs Sales Volume)
```python
# Plotting Price vs Sales Volume (Count of Sales)
price_vs_sales = car_data.groupby('Price ($)').size()

# Visualization
price_vs_sales.plot(kind='line', title="Price vs. Sales Volume", ylabel="Number of Sales", figsize=(10, 6))
plt.show()
```

### Car Model Popularity (Most Sold Models)
```python
# Most sold car models (Top 10)
top_models = car_data['Model'].value_counts().head(10)

# Visualization
top_models.plot(kind='barh', title="Top 10 Most Sold Car Models", xlabel="Count of Sales", figsize=(10, 6))
plt.show()
```

### Customer Segmentation (Clustering)
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = car_data[['Annual Income', 'Price ($)']]

# Standardizing features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
car_data['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualization of the clusters
sns.scatterplot(x='Annual Income', y='Price ($)', hue='Cluster', data=car_data, palette='viridis')
plt.title("Customer Segmentation - Annual Income vs Price")
plt.show()
```
## Insights, Findings, and Recommendations

#### **Insights**
- Monthly sales trends show seasonality, with peaks during holidays and promotions.
   
- Top 10 car models by price contribute significantly to overall revenue.  

- Higher-income regions favor premium models, while lower-income regions prefer budget options.  

- Sales drop significantly for models priced above a certain threshold, highlighting price sensitivity.  

- Customer clusters reveal distinct preferences based on income and spending patterns.  

#### **Findings**  

- Mid-range models sell the most, while high-end and low-end models cater to niche markets.  

- Sales are concentrated in high-income regions with better infrastructure.  

- Seasonal sales spikes align with key demand periods.  

- High-performing models benefit from consistent pricing and brand trust.  

#### **Recommendations**  

- Stock mid-range models before seasonal demand spikes.  

- Target premium car promotions to high-income regions; offer discounts in low-income areas.  

- Optimize pricing strategies to address customer price sensitivity.  

- Introduce loyalty programs and enhanced after-sale services for repeat customers.  

- Use predictive analytics to adjust inventory, pricing, and marketing strategies.
  
