import numpy as np
import pandas as pd
import xarray as xr

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

class PCRR(object):
    """Class that implements principal component ridge regression"""
    def __init__(self, n_comp=20):
        self.n_comp=n_comp
        
    def fit(self, X_train, y_train):
        # Fit the encoder
        self.encoder = PCA(n_components=self.n_comp)
        self.encoder.fit(X_train)

        # Encode training data
        self.X_train_encoded = self.encoder.transform(X_train)

        # Fit the regularized regressor
        self.regressor = RidgeCV(alphas=np.logspace(-5,5,100))
        self.regressor.fit(self.X_train_encoded, y_train)
        
    def predict(self, X_test):
        try:
            self.X_test_encoded = self.encoder.transform(X_test)
            self.predictions = self.regressor.predict(self.X_test_encoded)
        
        except AttributeError:
            print('Fit the model first.')


# Read in target data: climate indices 
idir = '/media/maffie/MAFFIE2TB/Projects/COI/ObservedIndices/'
idirncep = '/media/maffie/MAFFIE2TB/Projects/COI/NCEP/'
figdir = '/home/maffie/plots/coi/'

idir = './'
idirncep = './'
figdir = './'

indices = pd.read_csv(idir+'tele_index.nh_195001-201709.csv')

# Drop year and month columns, replace with a datetime index
# So that it corresponds with the time dimension in the .nc file with input data
time = pd.date_range(start='1950-01', end='2017-09', freq='MS')
indices = indices.drop(['yyyy', 'mm'], axis=1).set_index(time)

# Set NaNs
indices[indices == -99.90] = np.nan

# Drop columns that we don't need
indices = indices.drop(['Expl.Var.'], axis=1)

# Read in three datasets:
# 1. ncep_hgt_anom:  data used to calculate NOAA index
# 2 = reanalysis dataset, 1900-2010
# 3 = model output, 1976-2015

ncep_hgt_anom = xr.open_dataset('./data/ncep_hgt.mon.mean.z500_anom.nc')
ncep_hgt_anom

INDEX = 'NAO'
# Split in train and test data
y_train_raw = indices[INDEX][:'1999']
y_test_raw = indices[INDEX]['2000':'2016']

# Store location of missing entries to remove them further on
index_missing_train = y_train_raw.isnull()
index_missing_test = y_test_raw.isnull()

# Input data: standardized anomalies + absolute pressure values for plotting
x_train_raw = np.squeeze(np.array(ncep_hgt_anom['hgt'].sel(time=slice('1950','1999'))))
x_test_raw  = np.squeeze(np.array(ncep_hgt_anom['hgt'].sel(time=slice('2000','2016'))))

x_train_flat = np.reshape(x_train_raw, (x_train_raw.shape[0], -1))
x_test_flat = np.reshape(x_test_raw, (x_test_raw.shape[0], -1))


model = PCRR()
model.fit(x_train_flat,y_train)
model.predict(x_test_flat)

