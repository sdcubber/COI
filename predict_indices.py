# Do it for one index or for all indices?

import os
import argparse  # parse command line arguments
import pandas as pd
import xarray as xr

from PCRR import PCRR  # Custom model class
# 1. Train the model on the training observations


# 2. Make predictions


class data_class(object):
    """Class to hold all necessary data
    with methods to clean up the data"""

    def __init__(self, args):
        try:
            self.df = pd.read_csv(args.y_train)
        except:
            print('Provide path to valid .csv file with indices')

        try:
            self.nc = xr.open_dataset(args.x_train)
        except:
            print('Provide path to valid .nc file with pressure anomalies')

    def cleanup_df(self):
        time = pd.date_range(start='1950-01', end='2017-09', freq='MS')
        # Drop year and month columns, replace with a datetime index
        # So that it corresponds with the time dimension in the .nc file with input data
        self.df = df.drop(['yyyy', 'mm'], axis=1).set_index(time)
        self.df[self.df == -99.90] = np.nan  # Set NaNs
        self.df = self.df.drop(['Expl.Var.'], axis=1)  # Drop columns that we don't need

    def get_y_train(self, coi='NAO'):
        pass

    def get_x_train(self):
        pass


def train_model(args):
    pass


def main():
    parser = argparse.ArgumentParser(
        description='Predict COIs with PCA-RR based on historical training data')
    parser.add_argument('x_train', type=str,
                        help='Path. Input data: .nc file with pressure anomalies')
    parser.add_argument('y_train', type=str,
                        help='Path. Input data: .csv file with observed indices')
    parser.add_argument('x_test', type=str,
                        help='Path. .nc file with inputs to make predictions')
    parser.add_argument('coi', type=str, default='NAO',
                        help='Index to predict')
    parser.add_argument('-nc', '--n_comp', type=int, default=20,
                        help='Number of principal components to use')

    args = parser.parse_args()

    print('Reading in the training data')
    data = data_class(args)

    x_train = data.get_x_train()
    y_train = data.get_y_train()
    x_test = data.get_x_test()

    print('Training the model...')
    model =


if __name__ == "__main__":
    main()
