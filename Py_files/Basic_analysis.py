import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from astropy.table import Table
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup

def try_float(x):
    '''Try to convert item to float, if that fails, leave it as the type that it is, likely a string
    -------------
    Parameters
    -------------
    x : type = variable - item to be converted to float if possible
    
    Returns
    -------------
    item passed as argument, converted to float if it can be
    '''
    try:
        return float(x)
    except ValueError:
        return x


def gaussian_func(x, amplitude, xmean, stddev):
    '''classic gaussian profile

    -------------

    Parameters
    -------------
    x :  type = float - value to be passed on the x-axis to get a y-axis value
    amplitude :  type = float - maximum height of the gaussian
    xmean : type = float - line center
    stddev : type = float - standard deviation of the gaussian
    
    Returns
    -------------
    A single y-value based on the given x-value and other parameters
    '''
    return (amplitude * np.exp(-0.5 * ((x - xmean) / stddev)**2))


def voigt(x, amp, center, sigma, gamma):
    '''classic voigt profile

    -------------

    Parameters
    -------------
    x :  type = float - value to be passed on the x-axis to get a y-axis value
    amp :  type = float - maximum height of the voigt
    center : type = float - line center
    sigma : type = float - standard deviation of the Gaussian contribution
    gamma : type = float - Full Width Half Max of the Lorenzian contribution

    Returns
    -------------
    A single y-value based on the given x-value and other parameters
    '''
    profile = voigt_profile(x - center, sigma, gamma)
    return amp * profile / np.max(profile)


import numpy as np


def get_continuum_around(wavelength_array, flux_array, feature_index, window_size = 25):
    '''Calculates the surrounding continuum around a feature.

    -------------
    Parameters
    -------------
    wavelength_array : array - array of wavelengths
    flux_array : array - array of flux values
    feature_index : int - center (or near center) index of the feature
    window_size (optional, defaults to 25) : type = int - number of indexes to search continuum for
    
    Returns
    -------------
    mean_cont : float - Average continuum flux value of surrounding wavelengths
    stdev : float - Standard deviation of the surrounding continuum (for noise estimate)
    '''

    n = len(wavelength_array)
    start = max(0, center_idx - window_size)
    end = min(n, center_idx + window_size + 1)
    
    # Extract flux values in the window
    window_fluxes = flux_array[start:end]
    
    # Calculate IQR-based outlier bounds
    q25, q75 = np.percentile(window_fluxes, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - iqr_mult * iqr
    upper_bound = q75 + iqr_mult * iqr
    
    # Filter out outliers
    filtered_fluxes = window_fluxes[(window_fluxes >= lower_bound) & (window_fluxes <= upper_bound)]
    
    return np.mean(filtered_fluxes), np.std(filtered_fluxes)


def fit_voigt_to(wavelength_of_feature, tolerance, wavelength_array, flux_array, type = True, u = 1e+6, show_plot = False, feature_idx_width = 6):
    '''Fits voigt profile to feature nearest to given wavelength.


    need to add backup trial

    -------------

    Parameters
    -------------
    wavelength_of_feature : type = float - wavelength closest 
    tolerance : type = float - number of units (u argument) that the center of the feature can be and still achieve a tag of 2
    wavelength_array :  type = float - array of wavelengths including features
    flux_array : type = list - flux density array at each corresponding wavelength
    type : type = boolean : True if emission feature, False if absorption
    u (optional, defaults to 1e+6 (microns)) : type = float - unit to convert to meters
    Show_plot (optional, defaults to false) : type = boolean - show plot of fit?
    feature_idx_width (optional, defaults to 6) : type = int - number of indexes on each side of the feature's center to fit to


    Returns
    -------------
    [xrange, fitted] : type = list - plotting datapoints in [x,y] format
    total_feature_flux : type = float - integrated flux in units of {flux}
    center_WL, 
    this_features_snr, 
    chi2, reduced_chi2, 
    [*params], 
    tag : type = boolean : 0 representing bad fit or feature not found, 1 representing decent fit, no warnings triggered
    '''   
    if len(wavelength) != len(flux_array):
        print(f'wavelength and flux array must be same length, instead len(wavelength) = {len(wavelength)}, but len(flux array) = {len(flux_array)}')
        return None
    
    voigt_func = [] #TJ initialize array of x,y data for each voigt function
    
    center_idx = np.argmin(np.abs(wavelengtharray - wavelength_of_feature)) #TJ assign the center index as the closest wavelength to the expected wavelength
    continuum, cont_std = get_continuum_around(wavelength_array, flux_array, center_idx) #TJ get continuum and continuum stddev
    idx_range = range(center_idx-np.floor(feature_idx_width/2),center_idx+np.ceil(feature_idx_width/2))
    plt_range = range(min(idx_range)-feature_idx_width, max(idx_range)+feature_idx_width)
    x_data = wavelength_array[idx_range] #TJ generate the x data as the 20 nearest datapoints
    y_data = flux_array[idx_range] - continuum #TJ correct y-data for the net above continuum
    flux_uncertainty = flux_unc[idx_range] #TJ assign uncertainty array
    # Initial guesses
    amp_guess = max(flux[center_idx-1:center_idx+1]-continuum) if type else min((flux[center_idx-1:center_idx+1]-continuum))
    mean_guess = wavelength_array[center_idx]
    width_guess = wavelength_array[center_idx+1] - wavelength_array[center_idx] if (1 > (wavelength_array[center_idx+1] - wavelength_array[center_idx]) > 0) else 0.001
    amp_bounds = [amp_guess*0.75, amp_guess*1.25] 
    
    bounds = ([min(amp_bounds), wavelength_array[center_idx-tolerance], 0, 0], [max(amp_bounds), wavelength_array[center_idx+tolerance], np.inf, np.inf])
    
    params, cov = curve_fit(voigt, x_data, y_data, p0=[amp_guess, mean_guess, width_guess, width_guess], bounds=bounds, maxfev=5000)

    xrange = np.linspace(min(wavelength_array[plt_range]),max(wavelength_array[plt_range]), len(wavelength_array[plt_range])*100) #TJ define high resolution xrange for plotting
    fitted = voigt(xrange, *params) #TJ create the fitted y-data
    total_feature_flux = np.trapz(fitted, xrange) #TJ integrate over fitted voigt to get total flux
    this_features_snr = params[0]/cont_std #TJ snr is just amp divided by the noise in continuum
    center_WL = params[1] #TJ assign center of the feature for redshift/velocity calculations
    this_feature_flux = flux[idx_range]
    this_features_unc = flux_unc[idx_range]
    residuals = this_feature_flux - voigt(WL[idx_range], *params)
    chi2 = np.sum((residuals / this_features_unc)**2)
    dof = len(y_data) - len(params)
    reduced_chi2 = chi2 / dof

    if this_features_snr > 4:
        tag = 1
    else:
        tag = 0
    voigt_func = [[xrange, fitted], total_feature_flux, center_WL, this_features_snr, chi2, reduced_chi2, [*params], tag]
    if show_plots:
        plt.plot(WL[plt_range], flux[plt_range]-continuum, label='Continuum-Subtracted', color='purple')
        plt.axvline(x=WL[center_idx], label = f'provided feature wavelength {WL[center_idx]:.2f}')
        if tag == 1:
            plt.plot(xrange, fitted, color='blue', label=f'idx:{i} {transitions[i]}')
        else:
            plt.plot(xrange, fitted, color='red', label=f'idx:{i} {transitions[i]}')
        plt.legend()
        plt.show()
    return voigt_func

def get_feature_statistics(rest_wl_array, transitions):
    c = 2.99792458e+8
    fluxes = []
    center_wl = []
    velocities = []
    z_temp = []
    for i, feature in enumerate(voigts):
        fluxes.append(feature[1])
        center_wl.append(feature[2])
        rest = rest_wl_array[i]*(1e-6)
        obs = feature[2]*(1e-6)
        velocity = c*(obs-rest)/rest
        velocities.append(velocity)
        z_temp.append(((obs-rest)/rest))
    z = np.nanmedian(z_temp)
    return fluxes, center_wl, velocities, z
