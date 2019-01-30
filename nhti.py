#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:00:23 2019

@author: maffie
"""

import pandas as pd
import xarray as xr
import numpy as np
import os
import cartopy.crs as ccrs
from matplotlib import colors as colors
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.ioff()

#%%

# Pick an index to predict:
INDEX = 'POL' 

# NAO: North Atlantic Oscillation
# EA: East Atlantic Pattern
# WP: West Pacific Pattern
# EP/NP: EastPacific/ North Pacific Pattern 
# PNA: Pacific/ North American Pattern 
# EA/WR: East Atlantic/West Russia Pattern
# SCA: Scandinavia Pattern 
# TNH: Tropical/ Northern Hemisphere Pattern
# POL: Polar/ Eurasia Pattern
# PT: Pacific Transition Pattern 

indexList = ['NAO','EA','WP','EP/NP','PNA','EA/WR','SCA','TNH','POL','PT']
indexList = ['NAO']
#%%

# Read in target data: climate indices 
basedir = '/home/maffie/data/coi/'
basedir = '/media/maffie/MAFFIE2TB/Projects/COI'
idir = '{}/ObservedIndices/'.format(basedir)
figdir = '/home/maffie/plots/coi/nhti'

## Set netcdf input
ncInputSource = 'NCEP'

if ncInputSource == 'NCEP':
    netcdfInput = '{}/NCEP/hgt.mon.mean_stdanom.nc'.format(basedir)

#%%

## Read the data
    
#indices = pd.read_csv(idir+'tele_index.csv',sep=';') #.drop('Var.',axis=1)     ## NEW file
indices = pd.read_csv(idir+'tele_index.nh_195001-201709.csv')  ## OLD file

# Drop year and month columns, replace with a datetime index
# So that it corresponds with the time dimension in the .nc file with input data
time = pd.date_range(start='1950-01', end='2017-09', freq='MS')
indices = indices.drop(['yyyy', 'mm'], axis=1).set_index(time)

# Set NaNs
indices[indices == -99.90] = np.nan

# Drop columns that we don't need
indices = indices.drop(['Expl.Var.'], axis=1)
indices.head()

#%%

## Plot all indices

widths=1

fig, axes = plt.subplots(len(indices.columns.values),1,figsize=(16,14))
fig.subplots_adjust(bottom=0.15)

count = 0

for coi in list(indices.columns.values):
   
    ax = axes[count]
       
    ax.bar(indices[coi][indices[coi]<=0].index.values,indices[coi][indices[coi]<=0],width=widths, facecolor='steelblue', \
       alpha=.8, edgecolor='steelblue', lw=0.5)

    ax.bar(indices[coi][indices[coi]>0].index.values,indices[coi][indices[coi]>0],width=widths,facecolor='coral', \
       alpha=.8, edgecolor='coral', lw=0.5)

    ax.grid(linestyle='--')
    ax.set_ylabel(coi, fontsize=15, backgroundcolor="w")
    ax.set_xlabel('Months since 1950-01-01', fontsize=15, backgroundcolor="w")
    ax.xaxis.grid(True, which='both')

    count = count + 1
    
plt.savefig(figdir+'Observed_NHTI.jpg',dpi=100,bbox_inches='tight')

# Function to plot predictions
def plot_predictions(y_test, predictions, score, index, ncInputSource, figdir, title='Predicting test data'):

    plt.close('all')
    fig, ax = plt.subplots(figsize=(15,10))
    y_test.plot(ax=ax, color='steelblue', marker='o')
    pd.Series(predictions, index=y_test.index).plot(ax=ax, color='seagreen', ls='--', marker='.')
    ax.legend(['Test data', 'Predictions'], fontsize=18)
    ax.set_xlabel('Time').set_fontsize(18)
    ax.set_ylabel(index).set_fontsize(18)
    ax.set_title('{}, R²: {}'.format(title, np.round(score,2))).set_fontsize(18)
    print('{}: {}'.format(INDEX, np.round(score,2)))
    
    figtitle = "{}_{}_timeseries_PCAPCR.png".format(index,ncInputSource)
    fig.savefig(os.path.join(figdir, figtitle.replace("/", "")),dpi=150,bbox_inches='tight')

#%%

for INDEX in indexList:
    
    # Split in train and test data
    y_train_raw = indices[INDEX][:'1999']
    y_test_raw = indices[INDEX]['2000':'2016']
    
    # Store location of missing entries to remove them further on
    index_missing_train = y_train_raw.isnull()
    index_missing_test = y_test_raw.isnull()
    
    # Read in input data: standardized anomalies of 500 hPa geopotential height.
    data = xr.open_dataset(netcdfInput) 
    #dataabs = xr.open_dataset(idirncep+'hgt.mon.mean.box.1950-2017.nc') 
    
    # Do the same train-test split
    x_train_raw = np.squeeze(np.array(data['hgt'].sel(time=slice('1950','1999'))))
    x_test_raw  = np.squeeze(np.array(data['hgt'].sel(time=slice('2000','2016'))))
    
    x_train_flat = np.reshape(x_train_raw, (x_train_raw.shape[0], -1))
    x_test_flat = np.reshape(x_test_raw, (x_test_raw.shape[0], -1))
    
    
    ## Prediction: regularized PCR       
    n_comp=20
    pca = PCA(n_components=n_comp)
    
    train_month = y_train_raw.index.month
    test_month = y_test_raw.index.month
    
    n_train = len(y_train_raw)
    n_test = len(y_test_raw)
    
    x_train_pca = np.zeros((n_train,n_comp*12))
    x_test_pca = np.zeros((n_test, n_comp*12))
    
    for i,m in enumerate(np.unique(train_month)):
    
        # Aggregated seasonal data per month
        aggregated_data_train = np.concatenate([x_train_flat[train_month == np.unique(train_month)[i-1]],
                                                x_train_flat[train_month == np.unique(train_month)[i]],
                                                x_train_flat[train_month == np.unique(train_month)[i+1 if i < 11 else 0]]],
                                               axis=1)
        
        aggregated_data_test = np.concatenate([x_test_flat[test_month == np.unique(test_month)[i-1]],
                                               x_test_flat[test_month == np.unique(test_month)[i]],
                                               x_test_flat[test_month == np.unique(test_month)[i+1 if i < 11 else 0]]],
                                               axis=1)
           
        # Only monthly data (works much better)
        x_train_month = x_train_flat[train_month == np.unique(train_month)[i]]
        x_test_month = x_test_flat[test_month == np.unique(test_month)[i]]
        
        pca.fit(x_train_month)
        
        x_train_pca[train_month==m, n_comp*(m-1):(n_comp)*m] = pca.transform(x_train_month)
        x_test_pca[test_month==m, n_comp*(m-1):(n_comp)*m] = pca.transform(x_test_month)    
    
    
    # Now remove entries with missing target
    y_train = y_train_raw[~index_missing_train]
    y_test = y_test_raw[~index_missing_test]
    x_train_pca = x_train_pca[np.where(~index_missing_train)]
    x_test_pca = x_test_pca[np.where(~index_missing_test)]
    
    
    # Regularized PCR
    model = RidgeCV(alphas=np.logspace(start=-5, stop=5, num=1000))
    model.fit(x_train_pca, np.array(y_train))
    #print(model.alpha_)
    
    predictions = model.predict(x_test_pca)
    
    score = model.score(x_test_pca, np.array(y_test))
    
    # Plot the prediction
    plot_predictions(y_test, predictions, score, INDEX, ncInputSource, figdir, 'PCA per month - regularized PCR')
