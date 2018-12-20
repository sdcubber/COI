#!bin/bash

#===============================================================================
#
#          FILE:  CalcAnomalies.sh
#
#         USAGE:  bash CalcAnomalies.sh ~/shared/CMIP5/CESM1-BGC zg500hpa ~/shared/CMIP5/CESM1-BGC 1976 2005
#
#   DESCRIPTION:  Calculate (standardized) anomalies of a climate input field
#
#     ARGUMENTS:  
#		1) Input directory, 
#			eg. ~/shared/CMIP5/CESM1-BGC
#		2) Variable name 
#			eg. zg500hpa
#		3) Output directory
#			eg. ~/shared/CMIP5/CESM1-BGC
#		4) Start year to calculate climatology 
#			eg. 1976
#		5) End year to calculate climatology 
#			eg. 2005
#        AUTHOR:  M. Demuzere, matthias.demuzere@ugent.be
#       COMPANY:  Ghent University
#       VERSION:  1.0
#       CREATED:  25/10/2017
#      REVISION:  ---
#
#===============================================================================

idir=$1
ivar=$2
odir=$3
climstart=$4
climstop=$5

## Go to input dir
cd $idir

## Inputfile and extension
ifile=$ivar'_Amon_CESM1-BGC_historical_r1i1p1_185001-200512'
ext='.nc'

## Create temp directory
tmpdir=$idir/tmp
mkdir -p $tmpdir

## Calculate climatology and standard deviation
cdo -P 4 ymonmean -selyear,$climstart/$climstop $ifile$ext $tmpdir/ymonmean.nc
cdo -P 4 ymonstd -selyear,$climstart/$climstop $ifile$ext $tmpdir/ymonstd.nc

## Calculate anomalies
cdo -P 4 sub $ifile$ext $tmpdir/ymonmean.nc $odir/$ifile'_anom'$ext

## Calculate standardized anomalies
cdo -P 4 div $odir/$ifile'_anom'$ext $tmpdir/ymonstd.nc $odir/$ifile'_stdanom'$ext

echo 'Cleaning up ...'
rm -rf $tmpdir

echo 'Routine ready and file available here: '$odir/$ifile'_stdanom'$ext

