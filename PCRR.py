# Custom model class to do PCA ridge regression
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA


class PCRR(object):
    """Class that implements principal component ridge regression"""

    def __init__(self, n_comp=20):
        self.n_comp = n_comp

    def fit(self, X_train, y_train):
        # Fit the encoder
        self.encoder = PCA(n_components=self.n_comp)
        self.encoder.fit(X_train)

        # Encode training data
        self.X_train_encoded = self.encoder.transform(X_train)

        # Fit the regularized regressor
        self.regressor = RidgeCV(alphas=np.logspace(-5, 5, 1000))
        self.regressor.fit(self.X_train_encoded, y_train)

    def predict(self, X_test):
        try:
            self.X_test_encoded = self.encoder.transform(X_test)
            self.predictions = self.regressor.predict(self.X_test_encoded)
            return(self.predictions)

        except AttributeError:
            print('Fit the model first.')

    def __str__(self):
        return 'Principal component ridge regression with {} components'.format(self.n_comp)
