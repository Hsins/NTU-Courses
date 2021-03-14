#!/bin/bash 
wget -O hw5_MF_model.hdf5 https://www.dropbox.com/s/74hehe527xbnjm3/hw5_MF_model.hdf5?dl=1
python hw5.py $1 $2 $3 $4
