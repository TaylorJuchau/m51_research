import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from pathlib import Path

import os
import glob
import re
import sys

from ipywidgets import interact, Dropdown

from photutils.aperture import CircularAperture, aperture_photometry
from spectral_cube import SpectralCube

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from astropy.wcs import WCS
from astropy.constants import c
from astropy.io import fits
from astropy.visualization import simple_norm, imshow_norm

import warnings
warnings.filterwarnings('ignore') #TJ ignore warnings (careful enabling this)

home_directory = "/d/ret1/Taylor/jupyter_notebooks/Research" 
parent_dir = Path(home_directory).resolve() #TJ current notebook's parent directory
os.chdir(parent_dir) #TJ change working directory to be the parent directory

from Py_files.Basic_analysis import * #TJ import basic functions from custom package

def load_and_sort_convolved_Karin_spectrum(file_path):
    '''import data and sort by wavelength from very particularly structured file
    -------------
    
    Parameters
    -------------
    file_path : type = str - path to file with data
    
    Returns
    -------------
    structured array ('wavelength', 'intensity', 'uncertainty') where intensity and uncertainty are in W/m2/Hz
    '''    
    with open(file_path, 'r') as file:
        header = file.readline().strip().split()
        #TJ check first line structure for compliance
        if ((len(header) == 3) & (type(try_float(header[0])) == type(0.1)) & (type(try_float(header[1])) == type(0.1)) & (type(try_float(header[2])) == type(0.1))):
            data_list = []
            data_list.append((try_float(header[0])*1e-6, try_float(header[1])*1e-20, try_float(header[2])*1e-20))
            aperture_area_sr = (np.pi * (((0.75*u.arcsec).to(u.rad))**2)).value
            for line in file:
                parts = line.strip().split(maxsplit=3)
                
                #TJ Convert numeric columns to floats
                wavelength = float(parts[0])*1e-6 #TJ float required for sorting
                intensity = try_float(parts[1])*1e-20 * aperture_area_sr
                uncertainty = try_float(parts[2])*1e-20 * aperture_area_sr
                
                data_list.append((wavelength, intensity, uncertainty))
        
            #TJ Define dtype with notes as string
            dtype = [
                ('wavelength', float),
                ('intensity', float),
                ('uncertainty', float),
            ]
            
            data = np.array(data_list, dtype=dtype)
            sorted_data = np.sort(data, order=['wavelength'])  #TJ Sort by wavelength
            
            return sorted_data
        else:
            print('''File format is not as expected. Should be 3 columns no header, if not, see "import_data_and_sort_by_wavelength"
            function from Flux_calibration notebook''')
            return None


def extract_filter_name(filename):
    '''extract entire filter name, for example: F164N
    -------------
    
    Parameters
    -------------
    file_name : type = str - name of filter's fits file of format similar to ngc5194_nircam_1v3_f164n_i2d.fits
        *note*: function keys on the "_" and .fits to get filter name, requires lower case filter names, see generalized "sort_filters" function
    
    Returns
    -------------
    filter name as string
    '''   
    # For .fits files: ngc5194_nircam_1v3_f164n_i2d.fits → "f164n"
    if filename.endswith('.fits'):
        parts = os.path.basename(filename).split('_')
        for part in parts:
            if part.startswith('f') and part[1:].replace('n', '').replace('w', '').replace('m', '').isdigit():
                return part.lower()
    # For .dat files: F070M.dat → "f070m"
    elif filename.endswith('.dat'):
        return os.path.splitext(os.path.basename(filename))[0].lower()
    return None


def generate_list_of_files():
    '''cross-matches files in filter_directory to filters with images in image_directory, sorts by filter number
    -------------
    
    Parameters
    -------------
    none 
    
    Returns
    -------------
    list of arrays, first entry is the image file array, second is the filter file array, both sorted by filter numer (in name)
    '''   
    filter_directory = '/d/crow1/tools/cigale/database_builder/filters/jwst/'
    path = ['nircam', 'miri']
    filter_files = np.concatenate([glob.glob(os.path.join(filter_directory + file_path, "*.dat")) for file_path in path])
    image_directory = 'Data_files/Image_files'
    image_files = glob.glob(os.path.join(image_directory, "*.fits"))
    # Initialize aligned lists
    image_file_array = []
    filter_file_array = []
    
    # Loop through .fits files and find matching .dat files
    for fits_file in image_files:
        fits_filter = extract_filter_name(fits_file)
        if not fits_filter:
            continue  # Skip if no filter name found
        
        # Search for matching .dat file
        for dat_file in filter_files:
            dat_filter = extract_filter_name(dat_file)
            if dat_filter == fits_filter:
                image_file_array.append(fits_file)
                filter_file_array.append(dat_file)
                break  # Stop searching after first match
    filter_name_array = [f.split("/")[-1] for f in filter_file_array] #TJ generate array of just the filter names
    filter_numbers = np.array([get_filter_number(file) for file in filter_name_array]) #TJ generate array of just filter numbers
    sort_indices = np.argsort(filter_numbers) #TJ sort by these numbers
    sorted_filter_names = np.array(filter_file_array)[sort_indices]
    sorted_image_files = np.array(image_file_array)[sort_indices]
    return sorted_image_files, sorted_filter_names

def get_filter_number(filter_name):
    '''extracts numbers from filter name (drops F, N, W, etc from the ends)
    -------------
    
    Parameters
    -------------
    filter_files : type = list - list of filter names
    
    Returns
    -------------
    filter name as string
    '''   
    match = re.search(r'[A-Za-z](\d+)[A-Za-z]', filter_name)  # Numbers between ANY letters
    return int(match.group(1)) if match else 0





def get_Fnu_transmission(Fnu_array, wl_array, transmission_array, trans_wl_array):
    '''get expected flux through filter in units of whatever the flux_array is. Make sure to convert to mks units
    -------------
    
    Parameters
    -------------
    Fnu_array : type = array - array of flux density values
    wl_array : type = array - array of wavelength values for the corresponding Fnu_array values (should be in meters)
    transmission_array : type = array - array of unitless transmission coefficient
    trans_wl_array : type = array - array of wavelength values for the corresponding transmission values (should be in meters)

    
    Returns
    -------------
    total_flux : type = float - in units of flux_array
    '''   
    if ((trans_wl_array[0] < wl_array[0]) or (trans_wl_array[-1] > wl_array[-1])): #TJ Check if wavelengths are compatible with filter
        return None
    #TJ convert all arrays to numpy arrays for better indexing
    Fnu_array = np.array(Fnu_array)
    wl_array = np.array(wl_array)
    transmission_array = np.array(transmission_array)
    trans_wl_array = np.array(trans_wl_array)

    
    #TJ Convert wavelength to frequency, reverse so freq increases left to right
    spec_freq_array = c / wl_array[::-1]
    Fnu_array = Fnu_array[::-1]
    trans_freq_array = c / trans_wl_array[::-1]
    transmission_array = transmission_array[::-1]

    #TJ Interpolate Fnu onto the transmission frequency grid
    #TJ this is because jwst transmission arrays are averages over BW widths which are much coarser than Fnu is.
    
    interp_Fnu = np.interp(trans_freq_array, spec_freq_array, Fnu_array)
    
    
    weight = transmission_array / trans_freq_array #TJ weight the numerator and denominator by T *d_nu over nu for integration
    numerator = np.trapz(interp_Fnu * weight, trans_freq_array)#TJ perform integration
    denominator = np.trapz(weight, trans_freq_array)
    ab_mean_flux = numerator / denominator
    # Numerator: Fν * Transmission / nu integrated over frequency
    
    return ab_mean_flux.value



def fake_missing_header_info(filepath):
    '''WARNING!!! This will overwrite blank header fields permanently. If you want to retain those empty fields,
        perform this function on a copy of the file
    -------------
    
    Parameters
    -------------
    filepath : type = str - string to location of IFU fits file
    
    Returns
    -------------
    nothing, updates the fits file to have the fields 'DATE-BEG' 'MJD-BEG' 'DATE-END' 'MJD-END' 'XPOSURE' 'TELAPSE'
    '''   
    
    with fits.open(filepath, mode='update') as hdul:
        hdr = hdul['SCI'].header
        
        # Dictionary of default values (key: (value, comment))
        defaults = {
            'DATE-BEG': ('2000-01-01T00:00:00', 'Default observation start date'),
            'MJD-BEG': (51544.0, 'Default MJD observation start'),
            'DATE-END': ('2000-01-01T00:01:00', 'Default observation end date'),
            'MJD-END': (51544.000694, 'Default MJD observation end'),
            'XPOSURE': (60.0, 'Default exposure time [s]'),
            'TELAPSE': (60.0, 'Default elapsed time [s]')
        }
        
        # Only add missing keywords
        for key, (value, comment) in defaults.items():
            if key not in hdr:
                hdr[key] = (value, comment)
                print(f"Added default {key} = {value}")
            else:
                print(f"Preserved existing {key} = {hdr[key]}")
        
        # Special case: If DATE-BEG exists but MJD-BEG doesn't, compute it
        if 'DATE-BEG' in hdr and 'MJD-BEG' not in hdr:
            try:
                t = Time(hdr['DATE-BEG'], format='isot')
                hdr['MJD-BEG'] = (t.mjd, 'Computed from DATE-BEG')
                print(f"Computed MJD-BEG from DATE-BEG: {t.mjd}")
            except ValueError:
                hdr['MJD-BEG'] = (51544.0, 'Fallback MJD value')
        
        # Similar for DATE-END/MJD-END
        if 'DATE-END' in hdr and 'MJD-END' not in hdr:
            try:
                t = Time(hdr['DATE-END'], format='isot')
                hdr['MJD-END'] = (t.mjd, 'Computed from DATE-END')
            except ValueError:
                pass



def get_IFU_spectrum(IFU_filepath, loc, radius):
    '''extract spectrum from IFU file with aperature of radius, centered at ra,dec = loc
    -------------
    
    Parameters
    -------------
    IFU_filepath : type = str - string to location of IFU fits file
    loc : type = list - ra, dec in degrees or SkyCoord object
    radius : type = float - radius of aperture, must have units attached (like u.deg or u.arcsecond)
    
    Returns
    -------------
    structured array with entries for "wavelength" and "intensity"
    '''   
    #fake_missing_header_info(IFU_filepath) #TJ run this if needed
    hdul = fits.open(IFU_filepath)
    header = hdul['SCI'].header
    wcs = WCS(header)
    cube = SpectralCube.read(IFU_filepath, hdu='SCI')

    # === CONVERT RA/DEC TO PIXEL COORDINATES ===
    # Create SkyCoord object for spatial coordinates
    if type(loc) == list:
        spatial_coords = SkyCoord(ra=loc[0]*u.deg, dec=loc[1]*u.deg)
    elif type(loc) == SkyCoord:
        spatial_coords = loc
    else:
        print('loc is not a list of ra, dec and it is not a SkyCoord object.')
        return None
    
    # Convert spatial coordinates to pixels
    x, y = wcs.celestial.all_world2pix(spatial_coords.ra.deg, 
                                      spatial_coords.dec.deg, 0)
    
    # === BUILD APERTURE ===
    if header['CDELT2'] != header['CDELT1']:
        print('pixels are not square! function revisit get_IFU_spectrum() function to fix')
        return None
    cdelt = np.abs(header['CDELT2']) * u.deg
    pixel_scale = cdelt.to(u.arcsec)  # arcsec/pixel
    pix_area = header['PIXAR_SR'] #TJ pixel area in steradians
    radius = radius.to(u.arcsec)
    radius_pix = (radius / pixel_scale).value
    aperture = CircularAperture((x, y), r=radius_pix)
    aperture_area_sr = np.pi * (radius.to(u.rad))**2

    # === CRITICAL UNIT HANDLING ===
    cube = cube.with_spectral_unit(u.m)  # Ensure wavelength in meters
    
    # Convert flux units properly
    # Step 1: MJy/sr → W/m²/Hz/sr
    cube = cube.to(u.W/(u.m**2 * u.Hz * u.sr))  
    
    # Step 2: Multiply by pixel area to get W/m²/Hz/pixel
    pix_area_sr = header['PIXAR_SR'] * u.sr
    cube = cube * pix_area_sr
    
    # Step 3: Perform aperture sum (now in W/m²/Hz)
    flux_density_spectrum = []
    for i in range(len(cube.spectral_axis)):
        image_slice = cube[i].value  # Now in W/m²/Hz
        phot = aperture_photometry(image_slice, aperture)
        flux_density_spectrum.append(phot['aperture_sum'][0])  # No extra multiplication!
    wavelengths = cube.spectral_axis.to(u.m).value
    flux_density_spectrum = np.array(flux_density_spectrum)

    dtype = [('wavelength', 'f8'), ('intensity', 'f8')]
    spectrum = np.zeros(len(cube.spectral_axis), dtype=dtype)
    spectrum['wavelength'] = cube.spectral_axis.to(u.m).value
    spectrum['intensity'] = np.array(flux_density_spectrum)

    return spectrum


def get_image_flux(image_file, loc, radius):
    '''extract flux from image file with aperature of radius, centered at ra,dec = loc
    -------------
    
    Parameters
    -------------
    image_file : type = str - string to location of image fits file
    loc : type = list - ra, dec in degrees or SkyCoord object
    radius : type = float - radius of aperture, must have units attached (like u.deg or u.arcsecond)
    
    Returns
    -------------
    flux_density observed through filter
    '''
    #TJ assign location as SkyCoord object
    if type(loc) == list:
        spatial_coords = SkyCoord(ra=loc[0]*u.deg, dec=loc[1]*u.deg)
    elif type(loc) == SkyCoord:
        spatial_coords = loc
    else:
        print('loc is not a list of ra, dec and it is not a SkyCoord object.')
        return None
    hdul = fits.open(image_file) #TJ load file
    data = hdul['SCI'].data*1e-20  #TJ convert flux density to mks units
    header = hdul['SCI'].header #TJ load header
    if header['BUNIT'] != 'MJy/sr': #TJ check if units are MJy/sr, output will be nonsensical if not
        print('flux is NOT in MJy/sr. review get_image_flux function to fix')
        return None
    pix_area = header["PIXAR_SR"] #TJ define the angular size of a pixel in staradians
    wcs = WCS(header) #TJ read in the world coordinate system
    radius_pixels = (radius).to_value(u.deg) / abs(header['CDELT2']) #TJ get the radius of the aperture in number of pixels
    
    #TJ Convert RA/Dec to pixel coordinates
    x, y = wcs.all_world2pix(spatial_coords.ra.deg, spatial_coords.dec.deg, 0)
    aperture = CircularAperture((x, y), r = radius_pixels)
    
    #TJ Perform aperture photometry
    phot_result = aperture_photometry(data, aperture)
    total_flux = phot_result['aperture_sum'][0]*pix_area  #TJ the result is in pixel units, multiply by steradians per pixel to get units right

    return total_flux    

def compare_IFU_to_image(IFU_filepath, image_filepath, filter_filepath, loc, radius):
    '''extract spectrum from IFU file with aperature of radius, centered at ra,dec = loc using filter data (must be in meters)
       extract total flux from image file for circle centered at loc with radius radius
       compare the two.
    -------------
    
    Parameters
    -------------
    IFU_filepath : type = str - string to location of IFU fits file
    image_filepath : type = str - string to location of image fits file
    filter_filepath : type = list - first entry is an array of wavelengths (in m)
    loc : type = list - ra, dec in degrees or SkyCoord object
    radius : type = float - radius of aperture, must have units attached (like u.deg or u.arcsecond)
    
    Returns
    -------------
    IFU predicted flux, image_extracted flux, ratio
    '''   
    IFU_hdul = fits.open(IFU_filepath)
    IFU_data = IFU_hdul['SCI'].data  # flux in MJy/sr or μJy/arcsec²
    IFU_header = IFU_hdul['SCI'].header
    IFU_SED_data = get_IFU_spectrum(IFU_filepath, loc, radius)
    aperture_area_sr = np.pi * (radius.to(u.rad))**2
    filter_data = []
    with open(filter_filepath, 'r') as f:
        header = f.readline().strip().split()
        for line in f:
            data_line = line.strip().split()
            filter_data.append(data_line)
            
    header, filter_T = filter_data[:2], np.array(filter_data[2:])
    filter_wl = [try_float(filter_T[i,0])*1e-10 for i in range(len(filter_T))]
    filter_trans = [try_float(filter_T[i,1]) for i in range(len(filter_T))]
    
    print('IFU', IFU_SED_data["wavelength"][0], IFU_SED_data["wavelength"][-1])
    print('filter', filter_wl[0], filter_wl[-1])
    
    IFU_expected_flux = ((get_Fnu_transmission(IFU_SED_data["intensity"], IFU_SED_data["wavelength"], filter_trans, filter_wl)))
    photo_flux = (get_image_flux(image_filepath, loc, radius))
    if IFU_expected_flux:
        return IFU_expected_flux, photo_flux, IFU_expected_flux/photo_flux
    else:
        return None
if __name__ == "__main__":
    #########################################################################
    #TJ update values in this region every time!

    include_karin = True
    show_raw_fluxes = True
    show_normalized_fluxes = True
    image_files, filter_files = generate_list_of_files()
    #SED_filepath = 'Data_files/ARM2_HII2_stitch.dat' #TJ switch to this one and rerun to use unconvolved array

    if include_karin:
        SED_filepath = 'Data_files/ARM2_HII2_conv_stitched_test.dat' 
        karin_SED_data = load_and_sort_convolved_Karin_spectrum(SED_filepath) #TJ this is a precomputed spectra to compare to IFU derived data
    #TJ define IFU file path
    IFU_filepath = 'Data_files/IFU_files/M51_SW_f290lp_g395m-f290lp_s3d.fits'
    IFU_hdul = fits.open(IFU_filepath)
    IFU_data = IFU_hdul['SCI'].data  # flux in MJy/sr or μJy/arcsec²
    IFU_header = IFU_hdul['SCI'].header
    
    loc = [202.4340450, 47.1732517] #TJ define location, radius, and aperture area
    radius = 0.75*u.arcsec
##############################################################################
    
    aperture_area_sr = np.pi * (radius.to(u.rad))**2
    
    print("Compressing IFU data cube into 2d array with flux summed over all pixels in aperture")
    IFU_SED_data = get_IFU_spectrum(IFU_filepath, loc, radius)
    
    print("Extracting filter data")
    filter_data_array = [] #TJ initialize filter data array
    
    for file in filter_files: #TJ loop through filter files and extract wl and transmission data
        data = []
        with open(file, 'r') as f:
                header = f.readline().strip().split()
                for line in f:
                    data_line = line.strip().split()
                    data.append(data_line)
                
        header, filter_T = data[:2], np.array(data[2:])

        wl = [try_float(filter_T[i,0])*1e-10 for i in range(len(filter_T))]
        T = [try_float(filter_T[i,1]) for i in range(len(filter_T))]
        filter_data_array.append([wl, T])
    #TJ initialize arrays
    if include_karin:
        Karin_expected_flux = []
    IFU_expected_flux = []
    photo_flux = []
    print('doing the work')
    for i, filter_data in enumerate(filter_data_array):
        if include_karin:
            Karin_expected_flux.append((get_Fnu_transmission(karin_SED_data["intensity"], karin_SED_data["wavelength"], filter_data[1], filter_data[0])).value)
        IFU_expected_flux.append((get_Fnu_transmission(IFU_SED_data["intensity"], IFU_SED_data["wavelength"], filter_data[1], filter_data[0])).value)
        photo_flux.append(get_image_flux(image_files[i], loc, radius))
    print("work completed")

    #TJ convert to numpy array for better indexing and mathematics
    #Karin_expected_flux = np.array(Karin_expected_flux)
    IFU_expected_flux = np.array(IFU_expected_flux)
    photo_flux = np.array(photo_flux)
    
    if show_normalized_fluxes:
        plt.style.use('seaborn-v0_8-paper')  #TJ just a random style for the plot
        
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12}) #TJ 
        # Now plot with sorted data
        plt.figure(figsize=(10, 6))
        if include_karin:
            plt.scatter([extract_filter_name(x) for x in filter_files], Karin_expected_flux/photo_flux, label='Karin', s=100, marker='o', color='blue')
        plt.scatter([extract_filter_name(x) for x in filter_files], IFU_expected_flux/photo_flux, label='IFU-determined', s=100, marker='x', color='red')
        
        plt.xticks(rotation=45, ha='right')
        plt.tick_params(axis='y', which='both', labelsize=10)
        plt.legend()
        plt.xlabel('Filter Names')
        plt.ylabel('Filter Pass Through (MJy)')
        plt.title("FLux transmitted through filter compared to spectrum-derived expectations \nNormalized to Spectrum-derived values")
        plt.tight_layout()
        plt.show()
    if show_raw_fluxes:
        plt.style.use('seaborn-v0_8-paper')  #TJ just a random style for the plot
        
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12}) #TJ 
        # Now plot with sorted data
        plt.figure(figsize=(10, 6))
        if include_karin:
            plt.scatter([extract_filter_name(x) for x in filter_files], Karin_expected_flux, label='Karin', s=100, marker='o', color='blue')
        plt.scatter([extract_filter_name(x) for x in filter_files], IFU_expected_flux, label='IFU-determined', s=100, marker='o', color='black')
        plt.scatter([extract_filter_name(x) for x in filter_files], photo_flux, label = 'image_extracted', s=100, marker = 'x', color = 'red')
        plt.xticks(rotation=45, ha='right')
        plt.tick_params(axis='y', which='both', labelsize=10)
        plt.legend()
        plt.xlabel('Filter Names')
        plt.ylabel('Filter Pass Through (MJy)')
        plt.title("FLux transmitted through filter compared to spectrum-derived expectations \nNormalized to Spectrum-derived values")
        plt.tight_layout()
        plt.show()
