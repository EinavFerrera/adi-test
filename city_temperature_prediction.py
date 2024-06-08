import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from polynomial_fitting import PolynomialFitting


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Load the data
    df = pd.read_csv(filename, parse_dates=['Date'])

    # Check the right values- temp
    df = df[df['Temp'] < 57]
    df = df[df['Temp'] > -15]

    # Add 'DayOfYear'
    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df


def israel_temp_and_std_plots(israel_df):
    # Plot 1-
    plt.figure(figsize=(14, 6))
    plt.title('Daily Temperature in Israel by Day of Year')
    plt.xlabel('Day of the Year')
    plt.ylabel('Temperature (°C)')
    sns.scatterplot(data=israel_df, hue='Year', x='DayOfYear', y='Temp', palette='tab10', legend='full')
    plt.legend(title='Year', fontsize='small', title_fontsize='medium', loc='upper right')
    plt.savefig(os.path.join('.', f"{'2'}.png"))
    plt.show()

    # Plot 2-
    # Group - Month and calculate std of Temp
    monthly_std = israel_df.groupby('Month')['Temp'].agg(['std'])
    # Columns in different color
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    plt.figure(figsize=(14, 6))
    bars = plt.bar(monthly_std.index, monthly_std['std'], color=colors)
    plt.title('Standard Deviation of Daily Temperatures by Month in Israel')
    plt.xlabel('Month')
    plt.ylabel('Temperatures Standard Deviation (°C)')
    plt.xticks(monthly_std.index)

    # Add bar values
    for bar in bars:
        y_value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y_value, round(y_value, 2), ha='center', va='bottom')
    plt.savefig(os.path.join('.', f"{'3'}.png"))
    plt.show()


def all_countries_month_ave(df):
    # Grope the days to months and add mean and std columns
    all_countries = df.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']}).reset_index()
    all_countries.columns = ['Country', 'Month', 'Temp_mean', 'Temp_std']

    # Plot each country
    plt.figure(figsize=(14, 6))
    for country in all_countries['Country'].unique():
        country_data = all_countries[all_countries['Country'] == country]
        plt.errorbar(country_data['Month'], country_data['Temp_mean'], yerr=country_data['Temp_std'], label=country,
                     capsize=3)

    plt.title('Countries Average Monthly Temperature')
    plt.xlabel('Month')
    plt.ylabel('Average Temperature (°C)')
    plt.legend(title='Country', fontsize='small', title_fontsize='medium', loc='upper right')
    plt.xticks(list(range(1, 13)))
    plt.savefig(os.path.join('.', f"{'4'}.png"))
    plt.show()


def evaluate_polynomial_models(israel_df):
    # Initialize X to be day of the year and y to temperature
    X, y = israel_df.DayOfYear, israel_df.Temp

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6, test_size=0.25)

    # Store the errors value to each k
    test_errors = {}

    # Evaluate polynomial models of degree 1 to 10
    for k in range(1, 11):
        polynomial_model = PolynomialFitting(k)
        polynomial_model.fit(X_train, y_train)
        test_error_to_k = round(polynomial_model.loss(X_test, y_test), 2)
        test_errors[k] = test_error_to_k
        print(f'Test error for polynomial degree {k}: {test_error_to_k}')

    # Plotting
    plt.figure(figsize=(14, 6))
    degrees = list(test_errors.keys())
    errors = list(test_errors.values())
    bars = plt.bar(degrees, errors, color='lightsteelblue')

    plt.xlabel('Polynomial Degree')
    plt.ylabel('Test Error')
    plt.title('Test Error for Polynomial Models of Different Degrees')
    plt.xticks(degrees)

    # Add bar values
    for bar, error in zip(bars, errors):
        y_value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y_value, error, ha='center', va='bottom')

    plt.savefig(os.path.join('.', f"{'5'}.png"))
    plt.show()

    # Find the best degree
    best_degree = min(test_errors, key=test_errors.get)

    return best_degree


def fit_and_evaluate_final_model(israel_df, best_degree):
    X, y = israel_df.DayOfYear, israel_df.Temp

    polynomial_model = PolynomialFitting(best_degree)
    polynomial_model.fit(X, y)

    countries_without_israel = df[df['Country'] != 'Israel']['Country'].unique()
    errors = {}
    for country in countries_without_israel:
        country_df = df[df['Country'] == country]
        X_country, y_country = country_df.DayOfYear, country_df.Temp
        error = polynomial_model.loss(X_country, y_country)  # Calculate the loss after we fit the israel df
        errors[country] = error

    colors = plt.cm.tab10(np.linspace(0, 1, len(countries_without_israel)))  # Colors for each country

    # Plot the errors
    plt.figure(figsize=(14, 6))
    bars = plt.bar(errors.keys(), errors.values(), color=colors)
    plt.xlabel('Country')
    plt.ylabel('Model Error')
    plt.title('Model Error for Each Country (Israel fit)')
    # Add bar values
    for bar in bars:
        y_value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y_value, round(y_value, 2), ha='center', va='bottom')

    plt.savefig(os.path.join('.', f"{'6'}.png"))
    plt.show()


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    # Filter the dataset to contain samples only from Israel
    israel_df = df[df['Country'] == 'Israel']
    israel_temp_and_std_plots(israel_df)

    # Question 4 - Exploring differences between countries
    all_countries_month_ave(df)

    # Question 5 - Fitting model for different values of `k`
    best_degree = evaluate_polynomial_models(israel_df)

    # Question 6 - Evaluating fitted model on different countries
    fit_and_evaluate_final_model(israel_df, best_degree)
