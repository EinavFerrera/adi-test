from typing import NoReturn
from linear_regression import LinearRegression
import numpy as np


class PolynomialFitting(LinearRegression):
    """
    Polynomial Fitting using Least Squares estimation
    """

    def __init__(self, k: int):
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        self.k = k
        self.linear_model = LinearRegression(False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # Transformed
        transformed_X = self.__transform(X)

        # Use the LinearRegression model to fit
        self.linear_model.fit(transformed_X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # Transformed
        transformed_X = self.__transform(X)

        # Use the LinearRegression model to predict
        prediction = self.linear_model.predict(transformed_X)
        return prediction

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        # Transformed
        transformed_X = self.__transform(X)

        # Use the LinearRegression model to predict
        loses = self.linear_model.loss(transformed_X, y)
        return loses

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        # Do the transformed with np.vander
        transformed_X = np.vander(X, N=self.k + 1, increasing=True)
        return transformed_X
