import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from linear_regression import LinearRegression


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    # Make a copy
    X_copy = X.copy()

    # Add the year of sale
    X_copy['year'] = X_copy['date'].str[:4].astype(float)

    # Drop ID and date
    X_copy.drop(['id', 'date'], axis=1, inplace=True)

    # Handle with nan values
    for column in X_copy.columns:
        X_copy[column] = X_copy[column].fillna(value=X_copy[column].mean())

    # Change the renovated year to build year if the house not renovated
    X_copy.loc[X_copy['yr_renovated'] == 0, 'yr_renovated'] = X_copy['yr_built']

    # Check if year the house build is before the house renovated, if note change the build year
    X_copy.loc[X_copy['yr_built'] > X_copy['yr_renovated'], 'yr_built'] = X_copy['yr_renovated']

    # Check if sqft lot is bigger than the sqft feet, if not change the sqft lot
    X_copy.loc[X_copy['sqft_lot'] < X_copy['sqft_living'], 'sqft_lot'] = X_copy['sqft_living']

    # Check if sqft lot 15 is bigger then the sqft feet15, if not change the sqft lot
    X_copy.loc[X_copy['sqft_lot15'] < X_copy['sqft_living15'], 'sqft_lot15'] = X_copy['sqft_living15']

    # Add ratio of living area to lot size.
    X_copy['living_area_ratio'] = X_copy['sqft_living'] / X_copy['sqft_lot']

    # Add the ratio of living area to the 15-nearest neighbors' living area
    X_copy['relative_living_area'] = X_copy['sqft_living'] / X_copy['sqft_living15']


    X = X_copy

    return X, y


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    # Make a copy
    X_copy = X.copy()

    # Add the year of sale
    X_copy['year'] = X_copy['date'].str[:4].astype(float)

    # Drop ID and date
    X_copy.drop(['id', 'date'], axis=1, inplace=True)

    # Handle with nan values
    for column in X_copy.columns:
        X_copy[column] = X_copy[column].fillna(value=X_copy[column].mean())

    # Change the renovated year to build year if the house not renovated
    X_copy.loc[X_copy['yr_renovated'] == 0, 'yr_renovated'] = X_copy['yr_built']

    # Check if year the house build is before the house renovated, if note change the build year
    X_copy.loc[X_copy['yr_built'] > X_copy['yr_renovated'], 'yr_built'] = X_copy['yr_renovated']

    # Check if sqft lot is bigger than the sqft feet, if not change the sqft lot
    X_copy.loc[X_copy['sqft_lot'] < X_copy['sqft_living'], 'sqft_lot'] = X_copy['sqft_living']

    # Check if sqft lot 15 is bigger then the sqft feet15, if not change the sqft lot
    X_copy.loc[X_copy['sqft_lot15'] < X_copy['sqft_living15'], 'sqft_lot15'] = X_copy['sqft_living15']

    # Add ratio of living area to lot size.
    X_copy['living_area_ratio'] = X_copy['sqft_living'] / X_copy['sqft_lot']

    # Add the ratio of living area to the 15-nearest neighbors' living area
    X_copy['relative_living_area'] = X_copy['sqft_living'] / X_copy['sqft_living15']

    X = X_copy

    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = "."):
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    y_standard_deviation = y.std()

    for column in X.columns:
        X_standard_deviation = X[column].std()

        covariance_x_y = np.cov(X[column], y)[0, 1]
        correlation = covariance_x_y / (X_standard_deviation * y_standard_deviation)

        plt.figure()
        plt.scatter(X[column], y, alpha=0.6)
        plt.title(f'{column} (Pearson Correlation: {correlation})')
        plt.xlabel(column)
        plt.grid(True)
        plt.ylabel('Response')
        plt.savefig(os.path.join(output_path, f'{column}.png'))
        plt.close()


def fit(X_train, y_train, X_test, y_test):
    losses_mean = []
    losses_std = []
    percentages = list(range(10, 101))  # 10% to 100%

    # Iterate over increasing percentages of the training set
    for pre in range(10, 101):
        losses = []

        for i in range(10):
            X_sampled = X_train.sample(frac=pre / 100, random_state=i)
            y_sampled = y_train.sample(frac=pre / 100, random_state=i)


            # create the linear model, fit and loss
            linear_model = LinearRegression()
            linear_model.fit(X_sampled, y_sampled)

            predict_y = linear_model.predict(X_test)
            loss = mean_squared_error(y_test, predict_y)
            losses.append(loss)

        # Calculate mean and std of losses
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        losses_mean.append(mean_loss)
        losses_std.append(2*std_loss)

    # Convert lists to numpy arrays for the plot
    losses_mean = np.array(losses_mean)
    losses_std = np.array(losses_std)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, losses_mean, label="Mean Loss", color="blue")
    plt.fill_between(percentages, losses_mean - losses_std, losses_mean + losses_std, color="blue", alpha=0.2,
                     label="Mean Loss +/- 2 STD")
    plt.xlabel('Percentage of Training Set (%)')
    plt.ylabel('Mean Loss')
    plt.title('Mean Loss as a Function of Training Set Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('.', f"{'1'}.png"))
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")

    # Drop the row if the price in nan
    df = df.dropna(subset=['price'])

    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.25)

    # Question 3 - preprocessing of housing prices train dataset
    X_train_pros, y_train_pros = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train_pros, y_train_pros)

    # Question 5 - preprocess the test data
    X_test_pros = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    fit(X_train_pros, y_train_pros, X_test_pros, y_test)

