import os
os.environ["STPSF_PATH"] = "/d/ret1/Taylor/stpsf-data/" 
import webbpsf
os.environ["STPSF_PATH"] = "/d/ret1/Taylor/stpsf-data/" #TJ for some reason this only works if you do this line twice... no idea why

print(os.path.exists(os.environ["STPSF_PATH"]))
print(os.environ["STPSF_PATH"]) #TJ check that this kernel has access to the filter files
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import sys
from astropy.io import fits
from astropy.visualization import simple_norm, imshow_norm
from ipywidgets import interact, Dropdown
from astropy.wcs import WCS
from astropy.constants import c
from photutils.aperture import CircularAperture, aperture_photometry
import astropy.units as u
from astropy.table import Table
from tabulate import tabulate
from pathlib import Path
from tqdm.notebook import tqdm
from astropy.convolution import convolve_fft, Gaussian2DKernel


#parent_dir = Path().resolve().parent #TJ current notebook's parent directory

os.chdir('/d/ret1/Taylor/jupyter_notebooks/Research') #TJ change working directory to be the parent directory
from Py_files.Basic_analysis import * #import basic functions from custom package
from Py_files.Image_vs_spectra import * 
#TJ import flux calibration functions (mainly compare_IFU_to_image(IFU_filepath, image_filepath, filter_filepath, loc, radius))



def get_filter_wl_range(filter):
    '''Use the filter files to determine what wavelength range we need for each filter
    -------------
    
    Parameters
    -------------
    filter : type = str - string describing the filter name (case sensitive), for example "F335M"

    Returns
    -------------
    Path to newly convolved file as a string
    '''   
    filter_files = ['/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F115W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F140M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F150W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F164N.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F182M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F187N.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F200W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F210M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F212N.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F250M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F300M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F335M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F360M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F405N.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F430M.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/nircam/F444W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F560W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F770W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F1000W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F1130W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F1280W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F1500W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F1800W.dat',
       '/d/crow1/tools/cigale/database_builder/filters/jwst/miri/F2100W.dat']
    filter_file = [filer_filepath for filer_filepath in filter_files if extract_filter_name(filer_filepath).upper() == filter][0]
    filter_data = []
    with open(filter_file, 'r') as f:
        header = f.readline().strip().split()
        for line in f:
            data_line = line.strip().split()
            filter_data.append(data_line)

    header, filter_T = filter_data[:2], np.array(filter_data[2:])
    filter_wl = [try_float(filter_T[i,0])*1e-10 for i in range(len(filter_T))]
    return filter_wl[0]*u.m, filter_wl[-1]*u.m



def convolve(IFU_fits_file, instrument, filter, output_file = None):
    '''Convolve an IFU cube to the PSF of the provided filter.
    -------------
    
    Parameters
    -------------
    IFU_fits_file : type = str - string to location of IFU fits that you want to convolve.
    instrument : type = str - Either "MIRI" or "NIRCam" depending on wavelength
    filter : type = str - string describing the filter name (case sensitive), for example "F335M"
    output_file (optional, defaults to use the IFU_file with _convolved_to_{filter}.fits) : type = str - name of the convolved file
    
    Returns
    -------------
    Path to newly convolved file as a string
    '''   

    IFU_hdul = fits.open(IFU_fits_file)
    header = IFU_hdul["SCI"].header
    wl1, wl2 = get_filter_wl_range(filter)
    cube = SpectralCube.read(IFU_fits_file, hdu='SCI')
    if ((cube.spectral_axis[0] < wl1) & (cube.spectral_axis[-1] > wl2)):
        cube = cube.spectral_slab(wl1, wl2)
    
    spectral_axis = cube.spectral_axis  #TJ in meters
    # === Load webbpsf instrument ===
    if instrument == 'NIRCam':
        inst = webbpsf.NIRCam()
    elif instrument == "MIRI":
        inst = webbpsf.MIRI()
    inst.filter = filter

    # === Prepare output cube ===
    convolved_data = np.zeros_like(cube.unmasked_data[:].value)
    tqdm_kwargs = {
        'dynamic_ncols': True,  # Auto-adjusts width
        'mininterval': 0.5,     # Update every 0.5 seconds (optional)
        'position': 0,          # Fix position (set to 0 for notebooks)
        'leave': True           # Leaves progress bar after completion
    }
    # === Loop through wavelengths and convolve ===
    for i, wavelength in enumerate(tqdm(spectral_axis, desc=f"Convolving to {filter}")):
        psf = inst.calc_psf(monochromatic=wavelength.to(u.m).value)    
        psf_data = psf[0].data
        psf_data /= psf_data.sum()  # Normalize PSF to conserve flux
    
        image_slice = cube.filled_data[...].value[i]  # 2D image at this wavelength
        convolved_slice = convolve_fft(image_slice, psf_data, normalize_kernel=True, boundary='fill', fill_value=0)
        convolved_data[i] = convolved_slice
    
    # === Save the convolved cube ===
    out_hdu = fits.PrimaryHDU(convolved_data, header=header)
    if output_file:
        out_hdu.writeto(f"Data_files/IFU_files/{output_file}", overwrite=True)
        print(f"✅ PSF convolution complete and saved as {output_file}")
        return f"Data_files/IFU_files/{output_file}"
    else:
        out_hdu.writeto(f"Data_files/IFU_files/{IFU_fits_file.split('.f')[0]}_convolved_to_{filter}.fits", overwrite=True)
        print(f"✅ PSF convolution complete and saved as {IFU_fits_file.split('.f')[0]}_convolved_to_{filter}.fits")
        return f"Data_files/IFU_files/{IFU_fits_file}_convolved_to_{filter}.fits"

        