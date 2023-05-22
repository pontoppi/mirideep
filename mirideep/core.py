import pickle
import os
import warnings

import numpy as np
from scipy.signal import savgol_filter,correlate,medfilt
import matplotlib.pylab as plt
from matplotlib.patches import Circle

from astropy.modeling.models import BlackBody
from astropy.convolution import convolve, Gaussian1DKernel
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats

from photutils import aperture as ap
from photutils import centroids

from .utils import *

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
__version__ = 6.0

class MiriDeepSpec():

    def __init__(self,plot_centroid=False,shift_optimize=True,source='generic',save_intermediate=False,
                 bg_types={'ch1':'nod','ch2':'nod','ch3':'nod','ch4':'median'},
                 rrs={'ch1':1.4,'ch2':1.4,'ch3':1.4,'ch4':1.4},standard='athalia',wave_correct=True):
        self.local_path = os.path.join(os.path.dirname(__file__), 'rsrfs')
        self.standard = standard
        self.get_rsrf()
        self.plot_centroid = plot_centroid
        self.shift_optimize = shift_optimize
        self.source = source
        self.save_intermediate = save_intermediate
        self.wave_correct = wave_correct

        self.rrs = rrs
        self.bg_types = bg_types

    def run_extract(self):

        self.find_cubes()

        settings = {}
        waves = []
        spec1d_meds = []
        spec1d_stds = []
        spec1ds_intermediate = []
        rsrfs_intermediate = []
        ratios_intermediate = []
        waves_intermediate = []
        cens_intermediate = []

        for channel in ['1','2','3','4']:
            for band in ['short','medium','long']:
                dithers = [exposure for exposure in self.expdicts if exposure['channel']==channel if exposure['band']==band]
                setting = 'ch'+channel+'_'+band
                spec1ds = []
                for dither in dithers:
                    # which background to use? This was made because ch4 beams overlap in the 4-point dither
                    if self.bg_types['ch'+channel] == 'nod':
                        bg_cube = self.bg(dither,dithers)
                    else:
                        bg_cube = None

                    wave,spec1d,cen = self.extract(dither['file'],plot_centroid=self.plot_centroid,bg=bg_cube,rr=self.rrs['ch'+channel])

                    if (dither['channel'] in ['1']) and (dither['band'] in ['short','medium','long']):

                        rsrf_dither_indices = np.array([rsrf_dither['dither'] for rsrf_dither in self.rsrf_ch1[setting]])
                        rsrf_dither_index = np.where(rsrf_dither_indices == dither['dither'])
                        dither_rsrf = self.rsrf_ch1[setting][rsrf_dither_index[0][0]]['rsrf']
                        cen_rsrf = self.rsrf_ch1[setting][rsrf_dither_index[0][0]]['cen']

                        if self.shift_optimize:
                            rsrf_sh = self.shift_rsrf(wave,spec1d,dither_rsrf)
                        else:
                            rsrf_sh = dither_rsrf

                        model = self.standard_model(wave,standard='hd163466')
                        spec1d_defringe = spec1d/rsrf_sh * model

                        waves_intermediate.append(wave)
                        spec1ds_intermediate.append(spec1d)
                        rsrfs_intermediate.append((rsrf_sh/model)*np.nanmedian(spec1d)/np.nanmedian(rsrf_sh/model))
                        ratios_intermediate.append(spec1d_defringe)
                        cens_intermediate.append((cen_rsrf[0]-cen[0],cen_rsrf[1]-cen[1]))

                    else:
                        rsrf_dither_indices = np.array([rsrf_dither['dither'] for rsrf_dither in self.rsrf[setting]])
                        rsrf_dither_index = np.where(rsrf_dither_indices == dither['dither'])

                        # If the rsrf is missing dithers, set to the first one
                        if rsrf_dither_index[0].size == 0:
                            rsrf_dither_index = np.where(rsrf_dither_indices == 1)

                        dither_rsrf = self.rsrf[setting][rsrf_dither_index[0][0]]['rsrf']
                        cen_rsrf = self.rsrf[setting][rsrf_dither_index[0][0]]['cen']

                        if self.shift_optimize:
                            rsrf_sh = self.shift_rsrf(wave,spec1d,dither_rsrf)
                        else:
                            rsrf_sh = dither_rsrf

                        model = self.standard_model(wave,standard=self.standard)
                        spec1d_defringe = spec1d/rsrf_sh * model

                        waves_intermediate.append(wave)
                        spec1ds_intermediate.append(spec1d)
                        rsrfs_intermediate.append((rsrf_sh/model)*np.nanmedian(spec1d)/np.nanmedian(rsrf_sh/model))
                        ratios_intermediate.append(spec1d_defringe)
                        cens_intermediate.append((cen_rsrf[0]-cen[0],cen_rsrf[1]-cen[1]))

                    spec1ds.append(spec1d_defringe)

                #renormalize to median so that the sigma clip rejection works better
                med_all = np.nanmedian(np.stack(spec1ds).flatten())
                for ii,spec1d in enumerate(spec1ds):
                    spec1ds[ii] *= med_all/np.nanmedian(spec1d)

                spec1ds = np.stack(spec1ds)
                #spec1d_med = np.nanmedian(spec1ds,axis=0)
                stats = sigma_clipped_stats(spec1ds,axis=0,maxiters=5,sigma=2)
                spec1d_med = sigma_clipped_stats(spec1ds,axis=0,maxiters=3,sigma=2.0,grow=1)[0]
                spec1d_std = sigma_clipped_stats(spec1ds,axis=0,maxiters=1,sigma=5)[2]/2. #we could divide by 2 because we have 4 dithers.

                waves.append(wave)
                spec1d_meds.append(spec1d_med)
                spec1d_stds.append(spec1d_std)

        spec1d_meds = self.scale(waves,spec1d_meds)

        # Cut the low resolution end of overlapping segments
        for ii,wave in enumerate(waves[:-1]):
            ssubs = np.where(wave < np.min(waves[ii+1]))
            waves[ii] = waves[ii][ssubs]
            spec1d_meds[ii] = spec1d_meds[ii][ssubs]
            spec1d_stds[ii] = spec1d_stds[ii][ssubs]

        waves_flat = np.concatenate(waves)
        spec1d_flat = np.concatenate(spec1d_meds)
        spec1d_stds_flat = np.concatenate(spec1d_stds)
        ssubs = np.argsort(waves_flat)
        self.wave_all = waves_flat[ssubs]
        self.flux_all = spec1d_flat[ssubs]
        self.std_all = medfilt(spec1d_stds_flat[ssubs],31)

        if self.save_intermediate:
            with open(self.source+'_intermediates_v'+str(__version__)+'.npz', "wb") as pickleFile:
                pickle.dump({'waves':waves_intermediate,'spec1ds':spec1ds_intermediate,
                             'rsrfs':rsrfs_intermediate,'ratios':ratios_intermediate,'cens':cens_intermediate}, pickleFile)


        self.writespec(self.wave_all,self.flux_all,self.std_all,outname=self.source + '_1d_v' + str(__version__)+'.fits')

    def standard_model(self,wave,standard='jena'):
        if standard == 'jena':
            temp = 199*u.K
            scale = 5.77e8
            bb = BlackBody(temperature=temp)
            model = (bb(wave*u.micron) * scale).value
        if standard == 'athalia':
            temp = 195*u.K
            scale = 3.85e8
            bb = BlackBody(temperature=temp)
            model = (bb(wave*u.micron) * scale).value
        if standard == 'hd163466':
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
                    if bg_type=='nod':
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

    def get_rsrf(self):
        rsrf_file_ch1 = open(os.path.join(self.local_path,'hd163466_rsrf_6.0.npz'), 'rb')
        self.rsrf_ch1 = pickle.load(rsrf_file_ch1)
        rsrf_file_ch1.close()

        if self.standard=='athalia':
            rsrf_file = open(os.path.join(self.local_path,'athalia_rsrf_6.0.npz'), 'rb')
        elif self.standard=='jena':
            rsrf_file = open(os.path.join(self.local_path,'jena_rsrf_6.0.npz'), 'rb')
        else:
            print('Unknown standard')
            breakpoint()

        self.rsrf = pickle.load(rsrf_file)
        rsrf_file.close()

    def find_cubes(self,path='.'):
        datafiles = os.listdir(path)

        self.expdicts = []
        cubefiles = [datafile for datafile in datafiles if '_s3d.fits' in datafile]
        for cubefile in cubefiles:
            expdict = {}
            hdr = fits.getheader(cubefile)
            expdict['file'] = cubefile
            expdict['channel'] = hdr['CHANNEL']
            expdict['band'] = hdr['BAND'].lower()
            expdict['dither'] = hdr['PATT_NUM']
            expdict['pattern'] = hdr['PATTTYPE'].lower()
            if expdict['pattern'] != '4-point':
                raise ValueError('Only the 4-point dither pattern is currently supported')
            self.expdicts.append(expdict)

    def extract(self,cubefile,rr=1.7,plot_centroid=False,bg=None):

        cube = fits.getdata(cubefile)
        hdr = fits.getheader(cubefile,1)
        primary_hdr = fits.getheader(cubefile,0)

        if bg is not None:
            cube -= bg
        else:
            for ii in np.arange(cube.shape[0]):
                cube[ii,:,:] -= np.nanmedian(cube[ii,:,:])

        nw = cube.shape[0]
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

        coll = np.nanmedian(cube,axis=0)
        coll = np.nan_to_num(coll)
        coll_mask = np.ma.masked_less(coll,np.max(coll)/10.)
        cen = centroids.centroid_1dg(coll,mask=coll_mask.mask)

        spec1d = np.zeros(nw)

        for iw in np.arange(nw):
            plane = cube[iw,:,:]

            ap_radius = 1.22*rr*206265*wave[iw]/6.5e6 / cdelt1 / 3600
            aperture = ap.CircularAperture(cen,r=ap_radius)
            phot_table = ap.aperture_photometry(plane, [aperture])
            phot_val = phot_table['aperture_sum_0'][0]
            spec1d[iw] = phot_val * scale_factor # Units in Jy

        if self.plot_centroid:
            plt.imshow(coll)
            #circ = Circle((cen[0],cen[1]),ap_radius)
            #plt.patch(circ)
            plt.plot(cen[0],cen[1],marker='*',color='red')
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

    def scale(self,waves,spec1ds):

        nsegs = len(waves)
        for ii in np.arange(nsegs-1)+1:
            osubs_left = np.where(waves[ii-1]>np.min(waves[ii]))
            osubs_right = np.where(waves[ii]<np.max(waves[ii-1]))
            scale = np.nanmedian(spec1ds[ii-1][osubs_left])/np.median(spec1ds[ii][osubs_right])
            spec1ds[ii] *= scale
            print('scale:',scale)
            breakpoint()
        return spec1ds

    def bg(self,dither,dithers):

        cubes = []
        for bg_dither in dithers:
            #We can exclude the dither we are using from the bg estimation
            if bg_dither['file'] != dither['file']:
                cubes.append(fits.getdata(bg_dither['file']))

        bg_all = np.stack(cubes)
        bg_cube = np.nanmedian(bg_all, axis=0)
        return bg_cube

    def shift_rsrf(self,wave,spec1d,rsrf,maxlag = 7):

        spec1d_cont = savgol_filter(spec1d,int(spec1d.size/16.),2,mode='nearest')
        rsrf_cont = savgol_filter(rsrf,int(rsrf.size/16.),2,mode='nearest')

        corr = correlate(spec1d/spec1d_cont-np.mean(spec1d/spec1d_cont),rsrf/rsrf_cont-np.mean(rsrf/rsrf_cont),method='fft')
        lag =  np.argmax(corr[spec1d.size-maxlag:spec1d.size+maxlag]) - maxlag + 1

        model_gauss = models.Gaussian1D(amplitude=1., mean=maxlag+1, stddev=1.) + models.Linear1D(slope=0., intercept=0.)
        #fitter_gauss = fitting.LevMarLSQFitter()
        fitter_gauss = fitting.SLSQPLSQFitter()
        peakspec = corr[spec1d.size-maxlag:spec1d.size+maxlag]-np.min(corr[spec1d.size-maxlag:spec1d.size+maxlag])
        try:
            fit = fitter_gauss(model_gauss, np.arange(maxlag*2), peakspec, verblevel=0, maxiter=100)
            lag_fit = fit.mean_0.value - maxlag + 1
            #plt.plot(peakspec)
            #plt.plot(fit(np.arange(maxlag*2)))
            #plt.show()

            if fit.stddev_0.value>maxlag:
                print('Correlation peak too wide, using maximum value for lag: ', lag,fit.stddev_0.value)
                rsrf_sh = np.roll(rsrf,lag)
            else:
                print('Found correlation peak for lag: ', lag_fit,fit.stddev_0.value)
                rsrf_sh = np.interp(np.arange(rsrf.size)-lag,np.arange(rsrf.size),rsrf)
        except:
            print('Shift failed, assuming 0')
            rsrf_sh = rsrf

        return rsrf_sh

    def writespec(self,wave,fd,std,outname='spec1d.fits'):
        c1 = fits.Column(name='wavelength', array=wave, format='F')
        c2 = fits.Column(name='fluxdensity', array=fd, format='F')
        c3 = fits.Column(name='fluxdensity_stddev', array=std, format='F')

        t = fits.BinTableHDU.from_columns([c1, c2, c3])
        t.header['COMMENT'] = 'Processed by the JDISCS MIRI MRS pipeline v2'
        t.header['COMMENT'] = 'Klaus Pontoppidan'
        t.writeto(outname,overwrite=True)
