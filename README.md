# CBRAIN-CAM - a neural network climate model parameterization



Fork Author: Gunnar Behrens - <gunnar.behrens@dlr.de> 

Main Repository Author: Stephan Rasp - <raspstephan@gmail.com> - https://raspstephan.github.io

Thank you for checking out this fork of the CBRAIN repository (https://github.com/raspstephan/CBRAIN-CAM), dedicated to building VAEs for learning convective processes in SPCAM. This is a working repository, which means that the most current commit might not always be the most functional or documented. 


If you are looking for the exact version of the code that corresponds to the PNAS paper, check out this release: https://github.com/raspstephan/CBRAIN-CAM/releases/tag/PNAS_final [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1402384.svg)](https://doi.org/10.5281/zenodo.1402384)

For a sample of the SPCAM data used, click here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2559313.svg)](https://doi.org/10.5281/zenodo.2559313)


The modified climate model code is available at https://gitlab.com/mspritch/spcam3.0-neural-net. 
To create the training data (regular SPCAM) the correct branch is `fluxbypass`. To implement the trained neural network, check out `revision`.

### Papers using this fork

> G.Behrens, T. Beucler, P. Gentine, F. Iglesias-Suarez, M. Pritchard and V. Eyring, 2022.
> Non-linear dimensionality reduction with a Variational Autoencoder Decoder
> to Understand Convective Processes in Climate Models



### Papers using the main repository

> T. Beucler, M. Pritchard, P. Gentine and S. Rasp, 2020.
> Towards Physically-consistent, Data-driven Models of Convection.
> https://arxiv.org/abs/2002.08525

> T. Beucler, M. Pritchard, S. Rasp, P. Gentine, J. Ott and P. Baldi, 2019.
> Enforcing Analytic Constraints in Neural-Networks Emulating Physical Systems.
> https://arxiv.org/abs/1909.00912

> S. Rasp, M. Pritchard and P. Gentine, 2018.
> Deep learning to represent sub-grid processes in climate models.
> PNAS. https://doi.org/10.1073/pnas.1810286115
 
> P. Gentine, M. Pritchard, S. Rasp, G. Reinaudi and G. Yacalis, 2018. 
> Could machine learning break the convection parameterization deadlock? 
> Geophysical Research Letters. http://doi.wiley.com/10.1029/2018GL078202


## Repository description

The main components of the repository are:

- `cbrain`: Contains the cbrain module with all code to preprocess the raw data, run the neural network experiments and analyze the data.
- `environments`: Contains the .yml files of the conda environments used for this repository 
- `pp_config`: Contains configuration files and shell scripts to preprocess the climate model data to be used as neural network inputs
- `perf_analy`: Contains files for benchmarking the reproduction skills of evaluated models
- `R_2_val`: Contains saved data sets for the computation of values of determination R² of all networks
- `nn_config`: Contains neural network configuration files to be used with `run_experiment.py`.
- `notebooks`: Contains Jupyter notebooks used to analyze data. All plotting and data analysis for the papers of the main repository is done in the subfolder `presentation`. `dev` contains development notebooks.
- `saved_models`: Contains the saved models (weights, biases,..) of the VAE paper in respective subfolders. 
- `wkspectra`: Contains code to compute Wheeler-Kiladis figures. These were created by [Mike S. Pritchard](http://sites.uci.edu/pritchard/)
- `wk_spectrum`: Contains python code to compute Wheeler-Kiladis diagrams provided by Shuguang Wang.  
- `wheeler-kiladis`: Contains W-K diagrams shown in the VAE paper. 
- `save_weights.py`: Saves the weights, biases and normalization vectors in text files. These are then used as input for the climate model.
- `List_of_Figures.txt`: Contains a description where to find the python code to reproduce the figures of the VAE paper


