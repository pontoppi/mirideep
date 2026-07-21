"""
mirideep.core - Core spectral extraction and calibration module

This module contains the primary MiriDeepSpec class for extracting high signal-to-noise
1D spectra from JWST MIRI MRS observations. It performs advanced calibration beyond the
standard JWST pipeline including:

- Aperture photometry with diffraction-limited extraction
- Background estimation via nod subtraction or spatial annulus
- RSRF-based fringe removal with cross-correlation optimization
- Spectral segment stitching with flux scaling
- Wavelength corrections

Key Classes
-----------
MiriDeepSpec : Primary interface for spectral extraction
    Initialized with processing parameters (background methods, aperture radii, etc.)
    Main workflow via run_extract() method

Main Methods
------------
run_extract() : Complete extraction pipeline
    - Finds _s3d.fits cubes in working directory
    - Loads pre-computed RSRFs for each channel/band
    - For each dither: extracts spectrum, estimates background, applies RSRF correction
    - Combines dithers with sigma-clipping
    - Stitches spectral segments
    - Outputs FITS table with wavelength, flux, uncertainty, background

extract() : Aperture photometry on single cube
bg() : Background estimation (nod or annulus methods)
shift_rsrf() : Cross-correlation to optimize RSRF alignment
scale() : Flux scaling between overlapping spectral segments
writespec() : Write final 1D spectrum to FITS

Usage Example
-------------
>>> from mirideep.core import MiriDeepSpec
>>> md = MiriDeepSpec(source='mylup',
...                   bg_types={'ch1':'nod','ch2':'nod','ch3':'nod','ch4':'nod'},
...                   standard='jena2')
>>> md.run_extract()
# Outputs: mylup_1d_v9.5.fits

Author
------
Klaus Pontoppidan (klaus.m.pontoppidan@jpl.nasa.gov)

Version
-------
9.6 - Added intermediate diagnostic plotting (spectra and cross-correlation fits)
"""

import pickle
import os
import warnings
import copy

import numpy as np
from scipy.signal import savgol_filter,correlate,medfilt,find_peaks
from scipy.stats import norm
from scipy.ndimage import median_filter

import matplotlib.pylab as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages

from astropy.modeling.models import BlackBody
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel, Gaussian2DKernel
from astropy.convolution import interpolate_replace_nans
from astropy.convolution import Box2DKernel
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.time import Time

from photutils import aperture as ap
from photutils import centroids

from .utils import *

# Suppress ERFA dubious year warnings
from erfa import ErfaWarning
warnings.filterwarnings('ignore', category=ErfaWarning, message='.*dubious year.*')

# Suppress scipy RuntimeWarnings about empty slices
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')

# Suppress numpy RuntimeWarnings about degrees of freedom
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom <= 0 for slice.*')

# Suppress division warnings (expected when dividing by RSRF with zeros/nans)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in divide.*')

# Suppress astropy sigma clipping warnings about NaNs/infs (expected in raw spectral data)
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=AstropyWarning, message='.*Input data contains invalid values.*')

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
# Keep FITS file truncation warnings visible - these indicate data corruption
warnings.filterwarnings(action='default', message='File may have been truncated')
__version__ = 9.6

class MiriDeepSpec():
    '''
    Primary class for MIRI deep spectral extraction

    Input parameters:
    -------------------
    plot_centroid : bool
        Display centroid diagnostic plots
    shift_optimize : bool
        Optimize RSRF shift via cross-correlation
    source : str
        Source name for output files
    save_intermediate : bool
        Save intermediate products
    bg_types : dict
        Background method per channel {'ch1':'nod', 'ch2':'nod', ...}
    rrs : dict
        Aperture radii in diffraction limit units {'ch1':1.4, 'ch2':1.3, ...}
    standard : str or list of str
        Calibration source(s) for channels 2-4. If list, extracts with each and averages.
    ch1_standard : str or list of str
        Calibration source(s) for channel 1. If list, extracts with each and averages.
    wave_correct : bool
        Apply wavelength corrections
    single_shift : bool
        Use median shift for all dithers
    scale_rsrf : bool
        Apply amplitude scale factor to RSRF (default False)
    mask_ratio : float
        Ratio threshold for masking bad pixels
    source_cen : False or tuple
        (RA, Dec) for forced photometry, or False for auto-centroid
    scale_to_segment : False or int
        Renormalize scales to specific segment index

    Outputs:
    --------
    wave_all : array
        Wavelength array (microns)
    flux_all : array
        Flux density array (Jy)
    std_all : array
        Flux uncertainty array (Jy)
    bg_all : array
        Background array (MJy/sr)

    '''

    def __init__(self,plot_centroid=False,plot_fringematch=False,shift_optimize=True,source='generic',save_intermediate=False,
                 bg_types={'ch1':'nod','ch2':'nod','ch3':'nod','ch4':'nod'},
                 rrs={'ch1':1.4,'ch2':1.4,'ch3':1.4,'ch4':1.4},standard='jena2',ch1_standard='hd163466_COM',
                 wave_correct=True,single_shift=True,scale_rsrf=False,mask_ratio=20,centroid_type='1dg',
                 source_cen=False,scale_to_segment=False):

        # Input validation
        valid_channels = {'ch1', 'ch2', 'ch3', 'ch4'}
        valid_bg_methods = {'nod', 'annulus', 'fit'}

        # Validate bg_types
        if not isinstance(bg_types, dict):
            raise TypeError("bg_types must be a dictionary")
        for channel, method in bg_types.items():
            if channel not in valid_channels:
                raise ValueError(f"Invalid channel '{channel}' in bg_types. Must be one of {valid_channels}")
            if method not in valid_bg_methods:
                raise ValueError(f"Invalid background method '{method}' for {channel}. Must be one of {valid_bg_methods}")

        # Validate rrs
        if not isinstance(rrs, dict):
            raise TypeError("rrs must be a dictionary")
        for channel, radius in rrs.items():
            if channel not in valid_channels:
                raise ValueError(f"Invalid channel '{channel}' in rrs. Must be one of {valid_channels}")
            if not isinstance(radius, (int, float)):
                raise TypeError(f"Aperture radius for {channel} must be a number, got {type(radius)}")
            if radius <= 0:
                raise ValueError(f"Aperture radius for {channel} must be positive, got {radius}")

        # Validate that rrs and bg_types have matching channels
        if set(bg_types.keys()) != set(rrs.keys()):
            raise ValueError(f"bg_types and rrs must have matching channel keys. "
                           f"bg_types has {set(bg_types.keys())}, rrs has {set(rrs.keys())}")

        # Validate mask_ratio
        if not isinstance(mask_ratio, (int, float)) or mask_ratio <= 0:
            raise ValueError(f"mask_ratio must be a positive number, got {mask_ratio}")

        # Validate source_cen if provided
        if source_cen is not False:
            if not isinstance(source_cen, (tuple, list)) or len(source_cen) != 2:
                raise ValueError("source_cen must be False or a tuple/list of (RA, Dec)")
            if not all(isinstance(x, (int, float)) for x in source_cen):
                raise ValueError("source_cen coordinates must be numeric")

        # Validate scale_to_segment if provided
        if scale_to_segment is not False:
            if not isinstance(scale_to_segment, int) or scale_to_segment < 0:
                raise ValueError("scale_to_segment must be False or a non-negative integer segment index")

        # Convert standard and ch1_standard to lists if they aren't already
        if isinstance(standard, str):
            standard = [standard]
        if isinstance(ch1_standard, str):
            ch1_standard = [ch1_standard]

        # Validate standard and ch1_standard are lists
        if not isinstance(standard, list):
            raise TypeError("standard must be a string or list of strings")
        if not isinstance(ch1_standard, list):
            raise TypeError("ch1_standard must be a string or list of strings")

        self.local_path = os.path.join(os.path.dirname(__file__), 'rsrfs')
        self.standard = standard
        self.ch1_standard = ch1_standard
        self.plot_centroid = plot_centroid
        self.plot_fringematch = plot_fringematch
        self.shift_optimize = shift_optimize
        self.source = source
        self.save_intermediate = save_intermediate
        self.wave_correct = wave_correct
        self.single_shift = single_shift
        self.scale_rsrf = scale_rsrf
        self.mask_ratio = mask_ratio
        self.source_cen = source_cen
        self.centroid_type = centroid_type
        self.scale_to_segment = scale_to_segment

        # Dummy time values for figuring out the total observation duration
        self.exp_begin = Time('2050-01-01T00:00:00.0')
        self.exp_end = Time('2020-01-01T00:00:00.0')

        self.rrs = rrs
        self.bg_types = bg_types

    def run_extract(self):
        """
        Main extraction pipeline. If multiple calibrators are specified, extracts
        with each calibrator and averages the resulting spectra.
        """
        # find_cubes() only needs to be called once
        self.find_cubes()

        # Determine how many calibrator combinations we have
        n_standards = len(self.standard)
        n_ch1_standards = len(self.ch1_standard)
        n_calibrators = max(n_standards, n_ch1_standards)

        # If only one calibrator, use the original single extraction
        if n_calibrators == 1:
            self._extract_single(self.standard[0], self.ch1_standard[0])
        else:
            # Multiple calibrators - extract with each and average
            print(f"Extracting with {n_calibrators} calibrator combinations...")
            all_results = []

            for i in range(n_calibrators):
                # Use modulo to handle mismatched list lengths
                std = self.standard[i % n_standards]
                ch1_std = self.ch1_standard[i % n_ch1_standards]

                print(f"  Calibrator {i+1}/{n_calibrators}: standard='{std}', ch1_standard='{ch1_std}'")
                result = self._extract_single(std, ch1_std, return_results=True)
                all_results.append(result)

            # Average the results
            print("Averaging spectra from all calibrators...")
            self._average_results(all_results)

            # Write final averaged spectrum
            self.writespec(self.wave_all, self.flux_all, self.std_all, self.bg_all,
                          outname=self.source + '_1d_v' + str(__version__) + '.fits')

    def _extract_single(self, standard, ch1_standard, return_results=False):
        """
        Extract spectrum using a single calibrator combination.

        Parameters
        ----------
        standard : str
            Calibration source for channels 2-4
        ch1_standard : str
            Calibration source for channel 1
        return_results : bool
            If True, return the extracted spectrum arrays instead of storing them

        Returns
        -------
        dict or None
            If return_results is True, returns dict with wave_all, flux_all, std_all, bg_all
        """
        # Load RSRF for this specific calibrator combination
        self.get_rsrf(standard=standard, ch1_standard=ch1_standard)

        settings = {}
        waves = []
        spec1d_meds = []
        spec1d_stds = []
        bg1d_meds = []
        spec1ds_intermediate = []
        rsrfs_intermediate = []
        ratios_intermediate = []
        waves_intermediate = []
        cens_intermediate = []
        settings_intermediate = []
        xcorr_diagnostics_intermediate = []

        for channel in ['1','2','3','4']:
            for band in ['short','medium','long']:
                dithers = [exposure for exposure in self.expdicts if exposure['channel']==channel if exposure['band']==band]
                setting = 'ch'+channel+'_'+band

                # We may be using a different rsrf for channel 1
                if (channel in ['1']) and (band in ['short','medium','long']):
                    rsrf_dither_indices = np.array([rsrf_dither['dither'] for rsrf_dither in self.rsrf_ch1[setting]])
                else:
                    rsrf_dither_indices = np.array([rsrf_dither['dither'] for rsrf_dither in self.rsrf[setting]])

                spec1ds = []
                bg1ds   = []
                lags    = []
                
                for dither in dithers:
                    # which background to use? This was originally made because ch4 beams overlap in the 4-point extended dither pattern
                    bg_cube = self.bg(dither,dithers,self.bg_types['ch'+channel])

                    bg_cube_cp = copy.deepcopy(bg_cube)
                    nw = bg_cube.shape[0]
                    bg_1d = np.zeros(nw)
                    for ii in np.arange(nw):

                        plane = bg_cube_cp[ii,:,:]

                        #replace nans with median                        
                        plane[np.where(~np.isfinite(plane))] = np.nanmedian(plane)


                        #first half of values
                        mu, std = norm.fit(plane[(plane<np.percentile(plane, 80)) & (plane>np.percentile(plane,5))])
                        bg_1d[ii] = mu
                        
                    wave,spec1d,cen = self.extract(dither['file'],plot_centroid=self.plot_centroid,bg=bg_cube,rr=self.rrs['ch'+channel])
                    dither['wave'] = wave
                    dither['spec1d'] = spec1d
                    dither['cen'] = cen
                    dither['bg'] = bg_1d

                    rsrf_dither_index = np.where(rsrf_dither_indices == dither['dither'])
                    # If the rsrf is missing dithers, set to the first one
                    if rsrf_dither_index[0].size == 0:
                        rsrf_dither_index = np.where(rsrf_dither_indices == 1)

                    if (dither['channel'] in ['1']) and (dither['band'] in ['short','medium','long']):
                        dither_rsrf = self.rsrf_ch1[setting][rsrf_dither_index[0][0]]['rsrf']
                    else:
                        dither_rsrf = self.rsrf[setting][rsrf_dither_index[0][0]]['rsrf']

                    if self.shift_optimize:
                        if self.save_intermediate:
                            lag, diagnostics = self.shift_rsrf(wave,spec1d,dither_rsrf,return_diagnostics=True)
                            scale_factor = diagnostics['scale_factor']
                            xcorr_diagnostics_intermediate.append(diagnostics)
                        else:
                            lag, scale_factor = self.shift_rsrf(wave,spec1d,dither_rsrf)
                    else:
                        lag = 0
                        scale_factor = 1.0
                        if self.save_intermediate:
                            xcorr_diagnostics_intermediate.append(None)

                    lags.append(lag)
                    dither['scale_factor'] = scale_factor
                    print(f"  {setting:12s}  Dither {dither['dither']}  Lag: {lag:.3g}  Scale: {scale_factor:.3g}")

                #Find the best median lag per module
                lag_med = np.median(lags)
                lag_std = np.std(lags)
                print(f"{setting:12s}  Lag: {lag_med:.3g} +/- {lag_std:.3g}")

                for ii,dither in enumerate(dithers):

                    rsrf_dither_index = np.where(rsrf_dither_indices == dither['dither'])
                    # If the rsrf is missing dithers, set to the first one
                    if rsrf_dither_index[0].size == 0:
                        rsrf_dither_index = np.where(rsrf_dither_indices == 1)

                    if (dither['channel'] in ['1']) and (dither['band'] in ['short','medium','long']):
                        model = self.standard_model(wave,standard=ch1_standard)
                        dither_rsrf = self.rsrf_ch1[setting][rsrf_dither_index[0][0]]['rsrf']
                        cen_rsrf = self.rsrf_ch1[setting][rsrf_dither_index[0][0]]['cen']
                    else:
                        model = self.standard_model(wave,standard=standard)
                        dither_rsrf = self.rsrf[setting][rsrf_dither_index[0][0]]['rsrf']
                        cen_rsrf = self.rsrf[setting][rsrf_dither_index[0][0]]['cen']

                    if self.single_shift:
                        rsrf_sh = np.interp(np.arange(dither_rsrf.size)-lag_med,np.arange(dither_rsrf.size),dither_rsrf)
                    else:
                        rsrf_sh = np.interp(np.arange(dither_rsrf.size)-lags[ii],np.arange(dither_rsrf.size),dither_rsrf)

                    if self.scale_rsrf:
                        rsrf_cont = savgol_filter(rsrf_sh,int(rsrf_sh.size/24.),2,mode='nearest')
                        rsrf_sh = (rsrf_sh-rsrf_cont)*dither['scale_factor'] + rsrf_cont

                    spec1d_defringe = dither['spec1d']/rsrf_sh * model

                    if self.plot_fringematch:
                        # Check on fringe match. Was used for debugging. 
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.plot(wave,dither['spec1d'] / np.nanmedian(dither['spec1d']),label='raw')
                        ax.plot(wave,dither['spec1d']/rsrf_sh / np.nanmedian(dither['spec1d']/rsrf_sh),label='raw / rsrf')
                        ax.plot(wave,dither['spec1d']/rsrf_sh * model / np.nanmedian(dither['spec1d']/rsrf_sh * model), label='raw / rsrf * model')
                        ax.plot(wave,model/np.nanmedian(model), label='model')
                        ax.legend()
                        plt.show()

                    waves_intermediate.append(dither['wave'])
                    spec1ds_intermediate.append(dither['spec1d'])
                    rsrfs_intermediate.append((rsrf_sh/model)*np.nanmedian(dither['spec1d'])/np.nanmedian(rsrf_sh/model))
                    ratios_intermediate.append(spec1d_defringe)
                    cens_intermediate.append((cen_rsrf[0]-dither['cen'][0],cen_rsrf[1]-dither['cen'][1]))
                    settings_intermediate.append(setting)

                    spec1ds.append(spec1d_defringe)
                    bg1ds.append(dither['bg'])

                #renormalize to median so that the sigma clip rejection works better
                #Missing data should not fail, just be left out
                if len(spec1ds)>0:
                    med_all = np.nanmedian(np.stack(spec1ds).flatten())
                    bg_all = np.nanmedian(np.stack(bg1ds).flatten())
                    for ii,spec1d in enumerate(spec1ds):
                        spec1ds[ii] *= med_all/np.nanmedian(spec1d)
                        bg1ds[ii] *= bg_all/np.nanmedian(bg1ds[ii])

                    spec1ds = np.stack(spec1ds)
                    bg1ds = np.stack(bg1ds)

                    #spec1d_med = np.nanmedian(spec1ds,axis=0)
                    #stats = sigma_clipped_stats(spec1ds,axis=0,maxiters=5,sigma=2)
                    
                    spec1d_med = sigma_clipped_stats(spec1ds,axis=0,maxiters=3,sigma=2.,grow=False)[0]
                    spec1d_std = sigma_clipped_stats(spec1ds,axis=0,maxiters=1,sigma=5)[2]/2. #we divide by 2 because we have 4 dithers.
                    bg1d_med   = sigma_clipped_stats(bg1ds,axis=0,maxiters=3,sigma=2.,grow=False)[0]
                   
                    waves.append(wave)
                    spec1d_meds.append(spec1d_med)
                    spec1d_stds.append(spec1d_std)
                    bg1d_meds.append(bg1d_med)

        bg1d_meds,self.abs_flux_error_bg = self.scale(waves,bg1d_meds)
        spec1d_meds,self.abs_flux_error = self.scale(waves,spec1d_meds,silent=False)
        
        # Cut the low resolution end of overlapping segments
        for ii in np.arange(len(waves)-1):
            ssubs = np.where(waves[ii+1] > np.nanmax(waves[ii]))
            waves[ii+1] = waves[ii+1][ssubs]
            spec1d_meds[ii+1] = spec1d_meds[ii+1][ssubs]
            spec1d_stds[ii+1] = spec1d_stds[ii+1][ssubs]
            bg1d_meds[ii+1] = bg1d_meds[ii+1][ssubs]


        waves_flat = np.concatenate(waves)
        spec1d_flat = np.concatenate(spec1d_meds)
        spec1d_stds_flat = np.concatenate(spec1d_stds)
        bg1d_flat = np.concatenate(bg1d_meds)
        ssubs = np.argsort(waves_flat)
        wave_all = waves_flat[ssubs]
        flux_all = spec1d_flat[ssubs]
        std_all = medfilt(spec1d_stds_flat[ssubs],31)
        bg_all = bg1d_flat[ssubs]

        if return_results:
            # Return results for averaging
            return {
                'wave_all': wave_all,
                'flux_all': flux_all,
                'std_all': std_all,
                'bg_all': bg_all
            }
        else:
            # Single calibrator case - store and write directly
            self.wave_all = wave_all
            self.flux_all = flux_all
            self.std_all = std_all
            self.bg_all = bg_all

            if self.save_intermediate:
                with open(self.source+'_intermediates_v'+str(__version__)+'.npz', "wb") as pickleFile:
                    pickle.dump({'waves':waves_intermediate,'spec1ds':spec1ds_intermediate,
                                 'rsrfs':rsrfs_intermediate,'ratios':ratios_intermediate,'cens':cens_intermediate,
                                 'settings':settings_intermediate,'xcorr_diagnostics':xcorr_diagnostics_intermediate}, pickleFile)
                self.plot_intermediate_spectra(waves_intermediate, spec1ds_intermediate, rsrfs_intermediate, settings_intermediate)
                self.plot_cross_correlation(xcorr_diagnostics_intermediate, settings_intermediate)
            self.writespec(self.wave_all,self.flux_all,self.std_all,self.bg_all,outname=self.source + '_1d_v' + str(__version__)+'.fits')

    def _average_results(self, results_list):
        """
        Average spectra from multiple calibrator extractions.

        Parameters
        ----------
        results_list : list of dict
            List of extraction results, each containing wave_all, flux_all, std_all, bg_all
        """
        # Use the wavelength array from the first result (they should all be the same)
        self.wave_all = results_list[0]['wave_all']

        # Stack all flux arrays and average
        flux_stack = np.stack([r['flux_all'] for r in results_list])
        self.flux_all = np.nanmean(flux_stack, axis=0)

        # For uncertainties, it is still the mean since we expect to conservatively be dominated by the source not the standard
        std_stack = np.stack([r['std_all'] for r in results_list])
        self.std_all = np.nanmean(std_stack, axis=0)

        # Average backgrounds
        bg_stack = np.stack([r['bg_all'] for r in results_list])
        self.bg_all = np.nanmean(bg_stack, axis=0)

        print(f"Averaged {len(results_list)} spectra:")

    def standard_model(self,wave,standard='jena'):
        #nn = 6 # flatness of silicate feature
        #sil_cen = 10.2 # center of silicate feature
        #sil_amp = 0.035 # amplitude of silicate feature
        #sil_width = 1.5
        #eta = np.ones(wave.size)*0.9
        #eta += sil_amp * np.exp(-((wave - sil_cen) / sil_width)**nn)
        
        emissivity_scl = (self.emissivity['Emissivity']-0.78)/7.0+0.85 + 0.12*np.exp(-(self.emissivity['Wavelength']-29)**2/9**2)
        
        eta = np.interp(wave,self.emissivity['Wavelength'],emissivity_scl)

        if standard == 'jena':
            temp = 199*u.K
            scale = 5.77e8
            bb = BlackBody(temperature=temp)
            model = (bb(wave*u.micron) * scale).value * eta
        if standard == 'jena2':
            temp = 204*u.K
            scale = 1.16e9
            bb = BlackBody(temperature=temp)
            model = (bb(wave*u.micron) * scale).value * eta
        if standard == 'athalia':
            temp = 194*u.K
            scale = 5.6e8
            bb = BlackBody(temperature=temp)
            model = (bb(wave*u.micron) * scale).value * eta
        if standard == 'athalia2':
            temp = 206*u.K
            scale = 8.50e8
            bb = BlackBody(temperature=temp)
            model = (bb(wave*u.micron) * scale).value * eta
        if standard == 'athalia3':
            temp = 231*u.K
            scale = 10.3e8
            bb = BlackBody(temperature=temp)
            model = (bb(wave*u.micron) * scale).value * eta
        if 'hd163466' in standard:
            vsh = 0
            model_data = fits.getdata(os.path.join(self.local_path,'hd163466_mod_003.fits'),1)
            gauss_kernel = Gaussian1DKernel(100)
            model_conv = convolve(model_data['flux'], gauss_kernel)
            model_flux = model_conv*3.34e4*model_data['wavelength']**2
            model_wave = model_data['wavelength']/1e4 * (1+vsh/300000)
            model = np.interp(wave,model_wave,model_flux)

        return model


    def create_rsrf(self,standard='jena',bg_type='nod'):

        self.find_cubes()

        settings = {}

        for channel in ['1','2','3','4']:
            for band in ['short','medium','long']:
                dithers = [exposure for exposure in self.expdicts if exposure['channel']==channel if exposure['band']==band]

                for dither in dithers:
                    if self.bg_types['ch'+channel]:
                        bg_cube = self.bg(dither,dithers)
                        wave,spec1d,cen = self.extract(dither['file'],plot_centroid=self.plot_centroid,bg=bg_cube,rr=self.rrs['ch'+channel])
                    else:
                        wave,spec1d,cen = self.extract(dither['file'],plot_centroid=self.plot_centroid,bg=None,rr=self.rrs['ch'+channel])

                    dither['wave'] = wave
                    dither['rsrf'] = spec1d
                    dither['cen'] = cen

                setting = 'ch'+channel+'_'+band
                settings[setting] = dithers

        with open(standard+'_rsrf_'+str(__version__)+'.npz', "wb") as pickleFile:
            pickle.dump(settings, pickleFile)

    def get_rsrf(self, standard=None, ch1_standard=None):
        """
        Load RSRF calibration data for given standards.

        Parameters
        ----------
        standard : str, optional
            Standard for channels 2-4. If None, uses self.standard[0]
        ch1_standard : str, optional
            Standard for channel 1. If None, uses self.ch1_standard[0]
        """
        if standard is None:
            standard = self.standard[0]
        if ch1_standard is None:
            ch1_standard = self.ch1_standard[0]

        if ch1_standard=='hd163466_0823':
            rsrf_file_ch1 = open(os.path.join(self.local_path,'hd163466_0823_rsrf_9.5.npz'), 'rb')
        elif ch1_standard=='hd163466_0723':
            rsrf_file_ch1 = open(os.path.join(self.local_path,'hd163466_0723_rsrf_9.5.npz'), 'rb')
        elif ch1_standard=='hd163466_0624':
            rsrf_file_ch1 = open(os.path.join(self.local_path,'hd163466_0624_rsrf_9.5.npz'), 'rb')
        elif ch1_standard=='hd163466_COM':
            print("This option is deprecated")
            breakpoint()
        else:
            print('Unknown channel 1 standard')
            breakpoint()

        self.rsrf_ch1 = pickle.load(rsrf_file_ch1)
        rsrf_file_ch1.close()

        if standard=='athalia':
            rsrf_file = open(os.path.join(self.local_path,'athalia_rsrf_9.5.npz'), 'rb')
        elif standard=='athalia2':
            rsrf_file = open(os.path.join(self.local_path,'athalia2_rsrf_9.5.npz'), 'rb')
        elif standard=='athalia3':
            rsrf_file = open(os.path.join(self.local_path,'athalia3_rsrf_9.5.npz'), 'rb')
        elif standard=='jena2':
            rsrf_file = open(os.path.join(self.local_path,'jena2_rsrf_9.5.npz'), 'rb')
        elif standard=='jena':
            print("This option is deprecated")
            breakpoint()
            #rsrf_file = open(os.path.join(self.local_path,'jena_rsrf_8.0.npz'), 'rb')
        else:
            print('Unknown standard')
            breakpoint()

        self.rsrf = pickle.load(rsrf_file)
        rsrf_file.close()

        # Finally get asteroid emissivity spectrum
        self.emissivity = ascii.read(os.path.join(self.local_path,'emissivity.dat'))

    def find_cubes(self,path='.'):
        datafiles = os.listdir(path)

        self.expdicts = []
        cubefiles = [datafile for datafile in datafiles if '_s3d.fits' in datafile]
        exp_begins = []
        exp_ends   = []

        print(f"Found {len(cubefiles)} _s3d.fits files")

        for cubefile in cubefiles:
            expdict = {}
            try:
                # Check file integrity by attempting to open it
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    hdr = fits.getheader(cubefile)

                    # Check if truncation warning was raised
                    for warning in w:
                        if 'truncated' in str(warning.message).lower():
                            print(f"\n{'='*70}")
                            print(f"WARNING: {cubefile} appears to be truncated or corrupted!")
                            print(f"  {warning.message}")
                            file_size = os.path.getsize(cubefile)
                            print(f"  Actual file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                            print(f"  This file may need to be re-downloaded from MAST.")
                            print(f"  Skipping this file and continuing with remaining data...")
                            print(f"{'='*70}\n")
                            raise ValueError(f"Truncated file: {cubefile}")

            except (OSError, ValueError) as e:
                print(f"  Skipping {cubefile}: {e}")
                continue
            except Exception as e:
                print(f"  WARNING: Unexpected error reading {cubefile}: {e}")
                print(f"  Skipping this file.")
                continue

            expdict['file'] = cubefile
            expdict['channel'] = hdr['CHANNEL']
            expdict['band'] = hdr['BAND'].lower()
            expdict['dither'] = hdr['PATT_NUM']
            expdict['pattern'] = hdr['PATTTYPE'].lower()
            exp_begins.append(hdr['EXPSTART'])
            exp_ends.append(hdr['EXPEND'])

            if expdict['pattern'] != '4-point':
                raise ValueError('Only the 4-point dither pattern is currently supported')
            self.expdicts.append(expdict)

        self.exp_begin = np.min(exp_begins)
        self.exp_end   = np.min(exp_ends)
        self.exp_mid   = np.mean([self.exp_begin,self.exp_end])


    def extract(self,cubefile,rr=1.7,plot_centroid=False,bg=None,clean_nan=True):

        cube = fits.getdata(cubefile)
        hdr = fits.getheader(cubefile,1)
        primary_hdr = fits.getheader(cubefile,0)

        self.last_hdr = primary_hdr # Store the latest header read to global
        
        # subtrqct the background
        cube -= bg

        if clean_nan:
            cube[~np.isfinite(cube)] = 0.0

        nw = cube.shape[0]
        nx = cube.shape[2]
        ny = cube.shape[1]
        wave = (np.arange(nw))*hdr['CDELT3']+hdr['CRVAL3']

        # Correct for poor wavelength calibration in pipeline
        if self.wave_correct:
            wavecorr = fit_wavecorr(('ch'+primary_hdr['CHANNEL']+primary_hdr['BAND']).lower())
            wave += wavecorr(wave)

        midwave = wave[int(nw/2)]

        cdelt1 = hdr['CDELT1']
        cdelt2 = hdr['CDELT2']
        px_area = cdelt1*cdelt2 * 3.0461741978670859934e-4 #square degree --> steradian
        scale_factor = 1e6 * px_area # ---> Jy

        coll = np.nanmedian(cube[100:-100,:,:],axis=0)

        if self.source_cen:
            wcs = WCS(hdr)
            pix = wcs.wcs_world2pix(self.source_cen[0],self.source_cen[1],3,0)
            center = (pix[0],pix[1])
            cen = center
        else:        
            if self.centroid_type=='1dg':  
                coll[0:4,:] = 0
                coll[-4:,:] = 0 
                coll[:,0:4] = 0 
                coll[:,-4:] = 0 
                coll_mask = np.ma.masked_less(coll-np.nanmedian(coll),np.max(coll-np.nanmedian(coll))/self.mask_ratio)
                cen = centroids.centroid_1dg(coll,mask=coll_mask.mask)
            elif self.centroid_type=='max':

                coll_mask = np.ma.masked_less(coll-np.nanmedian(coll),np.max(coll-np.nanmedian(coll))/self.mask_ratio)                
                # indices of top 5 values
                nmax = 5
                coll[~np.isfinite(coll)] = 0
                max_indices = np.zeros(nmax, dtype=int)
                coll_tmp = copy.deepcopy(coll)
                ii = 0
                while ii<nmax:
                    max_indices[ii] = np.nanargmax(coll_tmp)
                    ymax,xmax = np.unravel_index(max_indices[ii], coll_tmp.shape)
                    coll_tmp[ymax,xmax] = 0
                    ii += 1
                    
                ymaxs = np.zeros(nmax)
                xmaxs = np.zeros(nmax)
                for ii,max_index in enumerate(max_indices):
                    ymaxs[ii],xmaxs[ii] = np.unravel_index(max_index, coll.shape)

                cen_coarse = (np.median(xmax),np.median(ymax))
                cens = centroids.centroid_sources(coll, cen_coarse[0], cen_coarse[1], box_size=9,centroid_func=centroids.centroid_2dg)
                cen = (cens[0][0],cens[1][0])

        spec1d = np.zeros(nw)

        for iw in np.arange(nw):
            plane = cube[iw,:,:]

            ap_radius = rr * self.difflimit(wave[iw],cdelt1)
            aperture = ap.CircularAperture(cen,r=ap_radius)

            phot_table = ap.aperture_photometry(plane, [aperture])
    
            phot_val = phot_table['aperture_sum_0'][0]
            spec1d[iw] = phot_val * scale_factor # Units in Jy

        if self.plot_centroid:
            plt.style.use(['dark_background'])
            fig = plt.figure(figsize=(5,4)) 

            ax = fig.add_subplot(111)
            im = ax.imshow(coll,cmap='magma',vmin=np.nanpercentile(coll,0.5),
                           vmax=np.nanpercentile(coll,99.5),origin='lower')
 
            circ = Circle((cen[0],cen[1]),ap_radius,edgecolor='orange',fill=False,lw=3)
            ax.add_patch(circ)
            ax.plot(cen[0],cen[1],marker='*',color='red')
            plt.show()
        
        # Interpolate nans
        bsubs = np.argwhere(np.isnan(spec1d))
        gsubs = np.argwhere(np.isfinite(spec1d))

        if np.any(bsubs):
            try:
                spec1d[bsubs.flatten()] = np.interp(wave[bsubs].flatten(),wave[gsubs].flatten(),spec1d[gsubs].flatten())
            except:
                spec1d[bsubs.flatten()] = np.nan

        return wave,spec1d,cen

    def difflimit(self,wave,pixsize):
        return 1.22*206265*wave/6.5e6 / pixsize / 3600

    def scale(self,waves,spec1ds,maxscale=0.1,silent=True):

        module = ['ch1 A','ch1 B','ch1 C','ch2 A','ch2 B','ch2 C','ch3 A','ch3 B','ch3 C','ch4 A','ch4 B','ch4 C']
        nsegs = len(waves)
        scales = np.ones(nsegs)
        for ii in np.arange(nsegs-1)+1:
            osubs_left = np.where(waves[ii-1]>np.min(waves[ii]))
            osubs_right = np.where(waves[ii]<np.max(waves[ii-1]))
            scale = np.nanmedian(spec1ds[ii-1][osubs_left])/np.nanmedian(spec1ds[ii][osubs_right])
            
            if ~np.isfinite(scale):
                scale = 1.0
            elif maxscale < scale < 1/maxscale:
                spec1ds[ii] *= scale
                if not silent:
                    print(module[ii]+': scale:',f'{(scale-1)*100:.3f}', '%')
                scales[ii] = scale
            else:
                if not silent:
                    print(module[ii]+': Calculated scaling factor out of bounds. Not scaling', f'{(scale-1)*100:.3f}', '%')
                scales[ii] = 1
            
        #Renormalize scale to avoid increasing uncertainty toward longer wavelengths
        if self.scale_to_segment:
            for ii in np.arange(nsegs):
                spec1ds[ii] /= scales[self.scale_to_segment]
        else:
            for ii in np.arange(nsegs):
                spec1ds[ii] /= np.nanmedian(scales)

        abs_flux_error = np.nanmean(np.abs(scales-1)*100)
        return spec1ds, abs_flux_error


    def bg(self,dither,dithers,bg_type='nod'):

        if bg_type=='nod':
            # The classic nod subtraction for low-background regions (most typical disk observations)

            cubes = []
            for bg_dither in dithers:
                #We can exclude the dither we are using from the bg estimation
                if bg_dither['file'] != dither['file']:
                    cubes.append(fits.getdata(bg_dither['file']))
            bg_all = np.stack(cubes)
            bg_cube = np.nanmedian(bg_all, axis=0)
      
        elif bg_type=='annulus':

            # Use an annulus to estimate the background level. Does not try to nod subtract. 
            # This is appropriate for high-background regions with a lot of spatial structure. 
            # Will not work well for multiple sources. 

            cube = fits.getdata(dither['file'])
            hdr  = fits.getheader(dither['file'],1)
            nw = cube.shape[0]

            # Make this a little bigger than the largest aperture size used for point sources. 
            ann_fac = 1.6 # inner radius of annulus in units of the diffraction limit
            
            # Standard wavelength scale
            wave = (np.arange(nw))*hdr['CDELT3']+hdr['CRVAL3']
            bg_spec_tot = np.nanmedian(cube,axis=(1,2))

            if self.source_cen:
                # We must know where the source is - we cannot trust an auto centroid in complex, high background regions. 
                wcs = WCS(hdr)
                pix = wcs.wcs_world2pix(self.source_cen[0],self.source_cen[1],3,0)
                cen = (pix[0],pix[1])
            else:
                print('The annulus background option only works with forced photometry with a user-designated source position')
                breakpoint()

            bg_spec_ann = np.zeros(nw)

            # We use the difference between a long and short wave plane to estimate the 2D structure of the background. 
            # Not perfect, but a lot better than a flat background. The wavelength reference is how many planes from the extreme ends of the cube. 
            wave_ref = 15
            nplanes = 50

            # How close to the source do we dare to get before using a flat background. 
            aper_radius = self.difflimit(np.max(wave),hdr['CDELT1']) * 1.6

            # Define all necessary apertures and annuli
            aperture = ap.CircularAperture(cen, aper_radius)
            annulus =  ap.CircularAnnulus(cen, r_in=aper_radius, r_out=aper_radius+3)
            short_stats_ann = ap.ApertureStats(np.nanmedian(cube[wave_ref:wave_ref+nplanes,:,:],axis=0), annulus)
            long_stats_ann = ap.ApertureStats(np.nanmedian(cube[-wave_ref-nplanes:-wave_ref,:,:],axis=0), annulus)

            # Calculate the peak flux of the point source (minus the background)
            short_stats = ap.ApertureStats(np.nanmedian(cube[wave_ref:wave_ref+nplanes,:,:],axis=0)-short_stats_ann.median, aperture)
            long_stats = ap.ApertureStats(np.nanmedian(cube[-wave_ref-nplanes:-wave_ref,:,:],axis=0)-long_stats_ann.median, aperture)

            # The 2D structure of the background is image(long)/F(source,long) - image(short)/F(source, short)
            bg_norm = np.nanmedian(cube[-wave_ref-nplanes:-wave_ref,:,:],axis=0)/long_stats.max - np.nanmedian(cube[wave_ref:wave_ref+nplanes,:,:],axis=0)/short_stats.max
            
            # normalizes to the annulus value of the 2D background
            bg_norm_stats = ap.ApertureStats(bg_norm, annulus)

            # Mask out the source residudals and replace by annulus median. 
            aperture_mask = aperture.to_mask(method='center')
            mask_data = aperture_mask.to_image(bg_norm.shape)
            bg_norm[mask_data > 0] = bg_norm_stats.median

            # Finally create the master background cube.
            bg_cube = np.ones(cube.shape)
            for iw in np.arange(nw):
                ann_radius = self.difflimit(wave[iw],hdr['CDELT1']) * ann_fac
                annulus = ap.CircularAnnulus(cen, r_in=ann_radius, r_out=ann_radius+3)
                bg_stats = ap.ApertureStats(cube[iw,:,:], annulus)
                bg_spec_ann[iw] = bg_stats.median
                
                bg_norm_stats = ap.ApertureStats(bg_norm, annulus)
                bg_cube[iw,:,:] = bg_norm * bg_stats.median/bg_norm_stats.median 

        elif bg_type=='fit':
            # Experimental algorithm where the background is estimated by a fit to the source and a background plane. 
            cube = fits.getdata(dither['file'])
            hdr  = fits.getheader(dither['file'],1)
            cube[np.where(~np.isfinite(cube))] = np.nanmedian(cube)
            bg_spec_tot = np.nanmedian(cube,axis=(1,2))

            nw = cube.shape[0]
            nx = cube.shape[2]
            ny = cube.shape[1]
            if self.source_cen:
                wcs = WCS(hdr)
                pix = wcs.wcs_world2pix(self.source_cen[0],self.source_cen[1],3,0)
                cen = (np.round(pix[0]).astype(int),np.round(pix[1]).astype(int))
            else:
                print('The fit background option only works with forced photometry with a user-designated source position')
                breakpoint()

            boxsize = 5

            yy, xx = np.mgrid[:boxsize*2, :boxsize*2]

            bg_spec_fit = np.zeros(nw)
            x_fit = np.zeros(nw)
            y_fit = np.zeros(nw)
            s_fit = np.zeros(nw)
            amp_fit = np.zeros(nw)

            for iw in np.arange(nw):
                diff_s = 3 
                edge_b = np.max([0,cen[1]-boxsize])
                edge_t = np.min([ny,cen[1]+boxsize])
                edge_l = np.max([0,cen[0]-boxsize])
                edge_r = np.min([nx,cen[0]+boxsize])

                cutout = cube[iw,edge_b:edge_t,edge_l:edge_r]
                yy, xx = np.mgrid[edge_b:edge_t, edge_l:edge_r]

                a_init = models.AiryDisk2D(amplitude=np.nanmax(cutout),radius=diff_s,x_0=boxsize,y_0=boxsize)
                c_init = models.Const2D(amplitude=np.nanmedian(cutout))
                fitter = fitting.LMLSQFitter()
                fitted = fitter(a_init+c_init,xx,yy,cutout)

                bg_spec_fit[iw] = fitted.amplitude_1.value
                x_fit[iw] = fitted.x_0_0.value
                y_fit[iw] = fitted.y_0_0.value
                s_fit[iw] = fitted.radius_0.value
                amp_fit[iw] = fitted.amplitude_0.value

            # Scaling full background spectrum to the fit level
            bg_scl = np.nanmedian(bg_spec_fit/bg_spec_tot)
            bg_cube = np.ones(cube.shape) * bg_spec_tot[:,np.newaxis,np.newaxis] * bg_scl


        elif bg_type=='median':
            # Just the median, please
            bg_spec = np.nanmedian(cube,axis=(1,2))
            bg_cube = np.ones(cube.shape) * bg_spec[:,np.newaxis,np.newaxis]
        else:
            print("Unknown background type", bg_type)
            breakpoint()
            
        return bg_cube

    def shift_rsrf(self,wave,spec1d,rsrf,maxlag = 19, return_diagnostics=False):

        spec1d_cont = savgol_filter(spec1d,int(spec1d.size/24.),2,mode='nearest')
        rsrf_cont = savgol_filter(rsrf,int(rsrf.size/24.),2,mode='nearest')

        corr1 = spec1d/spec1d_cont-np.mean(spec1d/spec1d_cont)
        stddev = np.std(corr1)
        bsubs = np.where((corr1>3*stddev) | (corr1<-3*stddev))
        corr1[bsubs] = 0
        corr2 = rsrf/rsrf_cont-np.mean(rsrf/rsrf_cont)
        stddev = np.std(corr1)
        bsubs = np.where((corr2>3*stddev) | (corr2<-3*stddev))
        corr2[bsubs] = 0

        corr1[~np.isfinite(corr1)] = 0
        corr2[~np.isfinite(corr2)] = 0

        corr = correlate(corr1,corr2,method='fft')
        lag =  np.argmax(corr[spec1d.size-maxlag:spec1d.size+maxlag]) - maxlag + 1

        # Calculate scale factor that minimizes std(corr1 - corr2_sh*scale_factor)
        # Optimal solution: scale_factor = sum(corr1 * corr2_sh) / sum(corr2_sh^2)
        corr2_sh = np.interp(np.arange(corr2.size)-lag, np.arange(corr2.size), corr2)
        valid = np.isfinite(corr1) & np.isfinite(corr2_sh)
        if np.sum(valid) > 0:
            numerator = np.sum(corr1[valid] * corr2_sh[valid])
            denominator = np.sum(corr2_sh[valid]**2)
            scale_factor = numerator / denominator if denominator != 0 else 1.0
        else:
            scale_factor = 1.0

        model_gauss = models.Gaussian1D(amplitude=np.max(corr), mean=maxlag+1, stddev=0.5)
        model_gauss.amplitude.min = 0
        model_gauss.amplitude.max = 1

        model_line  = models.Linear1D(slope=0., intercept=0.0,fixed={'slope':False,'intercept':False})
        model_total = model_gauss+model_line

        fitter_gauss = fitting.LevMarLSQFitter()
        #fitter_gauss = fitting.SLSQPLSQFitter()
        peakspec = corr[spec1d.size-maxlag:spec1d.size+maxlag]-np.min(corr[spec1d.size-maxlag:spec1d.size+maxlag])

        valleys = find_peaks(-peakspec)[0]
        
        try:
            #largest negative valley:
            valley_low = np.where(valleys - maxlag + 1 < 0, valleys, -np.inf).argmax()
            #smallers positive valley:
            valley_hi  = np.where(valleys - maxlag + 1 > 0, valleys, np.inf).argmin()

            #zero out areas outside of the main peak for stability
            peakspec[:valleys[valley_low]] = 0
            peakspec[valleys[valley_hi]:] = 0

            #Convolving the peak spectrum makes the fit much easier and more stable
            kernel = Gaussian1DKernel(stddev=2.5)
            peakspec = convolve(peakspec,kernel)

            fit = fitter_gauss(model_total, np.arange(maxlag*2), peakspec, maxiter=1000)
            lag_fit = fit.mean_0.value - maxlag + 1
            fit_succeeded = True

        except:
            print('cross correlation failed - no valid values. Assuming lag==0')
            lag_fit = 0
            fit = None
            fit_succeeded = False


        if return_diagnostics:
            # Return diagnostic data for plotting
            diagnostics = {
                'lag_fit': lag_fit,
                'maxlag': maxlag,
                'peakspec': peakspec,
                'fit': fit,
                'fit_succeeded': fit_succeeded,
                'corr1': corr1,
                'corr2': corr2,
                'wave': wave,
                'scale_factor': scale_factor
            }
            return lag_fit, diagnostics

        return lag_fit, scale_factor

    def writespec(self,wave,fd,std,bg,outname='spec1d.fits'):
        c1 = fits.Column(name='wavelength', array=wave, format='F', unit='micron')
        c2 = fits.Column(name='fluxdensity', array=fd, format='F', unit='Jy')
        c3 = fits.Column(name='fluxdensity_stddev', array=std, format='F', unit='Jy')
        c4 = fits.Column(name='background', array=bg, format='F', unit='MJy/sr')

        primary = fits.PrimaryHDU()
        t       = fits.BinTableHDU.from_columns([c1, c2, c3, c4])
        
        primary.header['DATE']     = (Time.now().isot, 'Time file was written by MIRIDeep')
        primary.header['COMMENT']  = 'Processed by the JDISCS MIRI MRS pipeline version '+str(__version__)
        primary.header['DOI']      = ('10.17909/tfk0-pa32','Digital Object Identifier')
        primary.header['HLSPLEAD'] = ('Klaus M. Pontoppidan','HSLP Principal Investigator')
        primary.header['HLSPID']   = ('JDISCS','HLSP Identifier')
        primary.header['HLSPNAME'] = ('JWST Disk Infrared Spectroscopic Chemistry Survey','HLSP project')
        primary.header['HLSPTARG'] = (self.last_hdr['TARGNAME'],self.last_hdr.comments['TARGNAME'])
        primary.header['HLSPVER']  = (__version__,'Data version')
        primary.header['LICENSE']  = ('CC BY 4.0','Data license')
        primary.header['LICENURL'] = ('https://creativecommons.org/licenses/by/4.0/','Data license URL')

        primary.header['PROPOSID'] = (self.last_hdr['PROGRAM'],self.last_hdr.comments['PROGRAM'])
        primary.header['VISIT_ID'] = (self.last_hdr['VISIT_ID'],self.last_hdr.comments['VISIT_ID'])
        primary.header['PI_NAME']  = (self.last_hdr['PI_NAME'],'Original program PI')
        primary.header['INSTRUME'] = ('MIRI','Instrument')
        primary.header['OBSERVAT'] = ('JWST','Observatory')
        primary.header['TELESCOP'] = ('JWST','Telescope')
        primary.header['DISPRSR']  = ('MRS','Dispersive element')
        primary.header['READPATT'] = (self.last_hdr['READPATT'],self.last_hdr.comments['READPATT'])

        primary.header['RADESYS']  = ('ICRS','Coordinate reference frame')
        primary.header['TARG_RA']  = (self.last_hdr['TARG_RA'],self.last_hdr.comments['TARG_RA'])
        primary.header['TARG_DEC'] = (self.last_hdr['TARG_DEC'],self.last_hdr.comments['TARG_DEC'])
        primary.header['SPECSYS']  = ('BARYCENT','Spectral reference frame')

        primary.header['TIMESYS']  = ('UTC','Code for time-related keywords')
        primary.header['XPOSURE']  = (self.last_hdr['EFFEXPTM']*self.last_hdr['NUMDTHPT'],'Total exposure time per sub-band')
        
        primary.header['DATE-BEG'] = (Time(self.exp_begin, format='mjd').isot,'Date-time start of exposures')
        primary.header['DATE-END'] = (Time(self.exp_end, format='mjd').isot,'Date-time end of exposures')
        primary.header['DATE-AVG'] = (Time(self.exp_mid, format='mjd').isot,'Date-time middle of exposures')
        
        primary.header['MJD-BEG']  = (self.exp_begin,'Start time of observation expressed as MJD')
        primary.header['MJD-END']  = (self.exp_end,'End time of observation expressed as MJD')
        primary.header['MJD-MID']  = (self.exp_mid,'Mid time of observation expressed as MJD')

        primary.header['CAL_VER']  = (self.last_hdr['CAL_VER'],self.last_hdr.comments['CAL_VER'])
        primary.header['CRDS_VER'] = (self.last_hdr['CRDS_VER'],self.last_hdr.comments['CRDS_VER'])
        primary.header['CRDS_CTX'] = (self.last_hdr['CRDS_CTX'],self.last_hdr.comments['CRDS_CTX'])

        # Format calibrator lists as comma-separated strings
        standard_str = ', '.join(self.standard) if isinstance(self.standard, list) else self.standard
        ch1_standard_str = ', '.join(self.ch1_standard) if isinstance(self.ch1_standard, list) else self.ch1_standard

        primary.header['STANDARD'] = (standard_str, 'RSRF standard(s)')
        primary.header['CH1_STAN'] = (ch1_standard_str, 'RSRF standard(s) for Channel 1')
        primary.header['NCALIB']   = (len(self.standard), 'Number of calibrators averaged')

        primary.header['ABSFLUXE'] = (f'{self.abs_flux_error:.3f}', 'Error on absolute flux (%)')


        hdulist = fits.HDUList([primary,t])
        hdulist.writeto(outname,overwrite=True)

    def plot_intermediate_spectra(self, waves, spec1ds, rsrfs, settings, poly_order=5, medfilt_size=21):
        """
        Plot spec1ds and rsrfs for each setting and dither position.

        Creates separate pages (figures) for each setting (channel-band combination)
        and saves them as pages in a single PDF file.
        Each page has 4 rows (one per dither) in landscape orientation.

        Parameters
        ----------
        waves : list of arrays
            Wavelength arrays for each dither
        spec1ds : list of arrays
            1D spectra for each dither
        rsrfs : list of arrays
            RSRF arrays for each dither
        settings : list of str
            Setting identifiers (e.g., 'ch1_short', 'ch2_medium') for each dither
        poly_order : int, optional
            Order of polynomial for continuum normalization (default=5)
        medfilt_size : int, optional
            Box size for median filter to make polynomial fits outlier-resistant (default=21)
        """
        # Get unique settings in order
        unique_settings = []
        for s in settings:
            if s not in unique_settings:
                unique_settings.append(s)

        # Create PDF file to hold all pages
        outname = self.source + '_intermediate_spectra_v' + str(__version__) + '.pdf'

        with PdfPages(outname) as pdf:
            # Create a separate page for each setting
            for setting in unique_settings:
                # Find all indices for this setting
                setting_indices = [i for i, s in enumerate(settings) if s == setting]
                n_dithers = len(setting_indices)

                # Calculate x-limits for this setting (95% of wavelength range)
                # Collect all wavelengths for this setting
                all_waves_setting = np.concatenate([waves[i] for i in setting_indices])
                wave_min = np.percentile(all_waves_setting[np.isfinite(all_waves_setting)], 2.5)
                wave_max = np.percentile(all_waves_setting[np.isfinite(all_waves_setting)], 97.5)

                # Create figure with landscape orientation: 1 column, 4 rows
                fig = plt.figure(figsize=(12, 10))

                for dither_idx, data_idx in enumerate(setting_indices):
                    wave = waves[data_idx]
                    spec1d = spec1ds[data_idx]
                    rsrf = rsrfs[data_idx]

                    # Normalize spec1d by fitting and dividing by polynomial
                    # Apply median filter to make fit outlier-resistant
                    finite_mask_spec = np.isfinite(spec1d) & np.isfinite(wave)
                    if np.sum(finite_mask_spec) > poly_order + 1:  # Need enough points for polynomial fit
                        spec1d_medfilt = medfilt(spec1d, kernel_size=medfilt_size)
                        poly_coeff_spec = np.polyfit(wave[finite_mask_spec], spec1d_medfilt[finite_mask_spec], poly_order)
                        poly_fit_spec = np.polyval(poly_coeff_spec, wave)
                        spec1d_norm = spec1d / poly_fit_spec
                    else:
                        spec1d_norm = spec1d / np.nanmedian(spec1d)

                    # Normalize rsrf by fitting and dividing by polynomial
                    # Apply median filter to make fit outlier-resistant
                    finite_mask_rsrf = np.isfinite(rsrf) & np.isfinite(wave)
                    if np.sum(finite_mask_rsrf) > poly_order + 1:
                        rsrf_medfilt = medfilt(rsrf, kernel_size=medfilt_size)
                        poly_coeff_rsrf = np.polyfit(wave[finite_mask_rsrf], rsrf_medfilt[finite_mask_rsrf], poly_order)
                        poly_fit_rsrf = np.polyval(poly_coeff_rsrf, wave)
                        rsrf_norm = rsrf / poly_fit_rsrf
                    else:
                        rsrf_norm = rsrf / np.nanmedian(rsrf)

                    # Create subplot - 4 rows, 1 column
                    ax = fig.add_subplot(4, 1, dither_idx + 1)

                    # Plot normalized spec1d (solid blue line)
                    ax.plot(wave, spec1d_norm, color='C0', alpha=0.8, linewidth=1.2,
                           label='spec1d')

                    # Plot normalized rsrf (dashed orange line)
                    ax.plot(wave, rsrf_norm, color='C1', alpha=0.7, linewidth=1.2,
                           linestyle='--', label='rsrf')

                    # Set x-limits to show 95% of wavelength range
                    ax.set_xlim(wave_min, wave_max)

                    # Determine y-limits by rejecting 5% outliers (2.5% on each side)
                    combined_data = np.concatenate([spec1d_norm[np.isfinite(spec1d_norm)],
                                                    rsrf_norm[np.isfinite(rsrf_norm)]])
                    if len(combined_data) > 0:
                        y_min = np.percentile(combined_data, 2.5)
                        y_max = np.percentile(combined_data, 97.5)
                        # Add 5% margin for better visualization
                        y_range = y_max - y_min
                        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

                    # Format subplot
                    ax.set_ylabel('Normalized Flux', fontsize=10)
                    ax.set_title(f'Dither {dither_idx+1}', fontsize=10, loc='left', fontweight='bold')
                    ax.grid(True, alpha=0.3, linestyle=':')
                    ax.legend(fontsize=9, loc='best', framealpha=0.9)

                    # Only show x-label and x-tick labels on bottom panel
                    if dither_idx == n_dithers - 1 or dither_idx == 3:
                        ax.set_xlabel('Wavelength (μm)', fontsize=10)
                    else:
                        ax.set_xlabel('')
                        ax.set_xticklabels([])

                # Add overall title
                fig.suptitle(setting.replace('_', ' ').upper(), fontsize=14, fontweight='bold', y=0.995)

                plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=0.5)

                # Save this page to the PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f"Saved intermediate spectra plot: {outname}")

    def plot_cross_correlation(self, xcorr_diagnostics, settings):
        """
        Plot cross-correlation fits for each setting and dither position.

        Creates separate pages (figures) for each setting (channel-band combination)
        and saves them as pages in a single PDF file.
        Each page has 4 rows (one per dither) in landscape orientation, showing
        the cross-correlation peak and its Gaussian fit.

        Parameters
        ----------
        xcorr_diagnostics : list of dict or None
            Cross-correlation diagnostic data for each dither (from shift_rsrf)
            Each dict contains: lag_fit, maxlag, peakspec, fit, corr1, corr2, wave
            None entries indicate shift_optimize was False for that dither
        settings : list of str
            Setting identifiers (e.g., 'ch1_short', 'ch2_medium') for each dither
        """
        # Get unique settings in order
        unique_settings = []
        for s in settings:
            if s not in unique_settings:
                unique_settings.append(s)

        # Create PDF file to hold all pages
        outname = self.source + '_xcorr_fits_v' + str(__version__) + '.pdf'

        with PdfPages(outname) as pdf:
            # Create a separate page for each setting
            for setting in unique_settings:
                # Find all indices for this setting
                setting_indices = [i for i, s in enumerate(settings) if s == setting]
                n_dithers = len(setting_indices)

                # Create figure with landscape orientation: 1 column, 4 rows
                fig = plt.figure(figsize=(12, 10))

                for dither_idx, data_idx in enumerate(setting_indices):
                    diag = xcorr_diagnostics[data_idx]

                    # Create subplot - 4 rows, 1 column
                    ax = fig.add_subplot(4, 1, dither_idx + 1)

                    if diag is not None:
                        # Extract diagnostic data
                        maxlag = diag['maxlag']
                        peakspec = diag['peakspec']
                        fit = diag['fit']
                        lag_fit = diag['lag_fit']

                        # Plot cross-correlation peak
                        lag_array = np.arange(maxlag*2) - maxlag + 1
                        ax.plot(lag_array, peakspec, 'o-', color='C0', alpha=0.7,
                               markersize=4, linewidth=1.5, label='Cross-correlation')

                        # Plot Gaussian fit
                        ax.plot(lag_array, fit(np.arange(maxlag*2)), '-', color='C1',
                               linewidth=2.0, label='Gaussian fit')

                        # Mark the fitted lag with a vertical line
                        ax.axvline(lag_fit, color='C2', linestyle='--', linewidth=1.5,
                                  alpha=0.7, label=f'Lag = {lag_fit:.2f}')

                        # Format subplot
                        ax.set_ylabel('Correlation', fontsize=10)
                        ax.set_title(f'Dither {dither_idx+1} (lag={lag_fit:.2f})',
                                    fontsize=10, loc='left', fontweight='bold')
                        ax.grid(True, alpha=0.3, linestyle=':')
                        ax.legend(fontsize=9, loc='best', framealpha=0.9)

                    else:
                        # No cross-correlation data (shift_optimize was False)
                        ax.text(0.5, 0.5, 'No cross-correlation\n(shift_optimize=False)',
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=12, color='gray')
                        ax.set_ylabel('Correlation', fontsize=10)
                        ax.set_title(f'Dither {dither_idx+1}', fontsize=10, loc='left',
                                    fontweight='bold')
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Only show x-label and x-tick labels on bottom panel
                    if dither_idx == n_dithers - 1 or dither_idx == 3:
                        ax.set_xlabel('Lag (pixels)', fontsize=10)
                    else:
                        ax.set_xlabel('')
                        if diag is not None:
                            ax.set_xticklabels([])

                # Add overall title
                fig.suptitle(setting.replace('_', ' ').upper() + ' - Cross-Correlation Fits',
                            fontsize=14, fontweight='bold', y=0.995)

                plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=0.5)

                # Save this page to the PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f"Saved cross-correlation fits plot: {outname}")
