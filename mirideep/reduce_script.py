"""
mirideep.reduce_script - JWST pipeline execution and data download

This module provides functions to download JWST MIRI MRS observations from MAST
and execute calibration pipeline stages 1-3. It automates the process of:

- Downloading rate files (_rate.fits) or uncalibrated files (_uncal.fits) from MAST
- Creating association files for pipeline processing
- Running calwebb_detector1 (optional, if starting from uncal files)
- Running calwebb_spec2 per channel to create 3D IFU cubes
- Running calwebb_spec3 to create combined products

The output _s3d.fits cubes are the input to mirideep.core.MiriDeepSpec for
high S/N extraction.

Key Functions
-------------
reduce() : Main pipeline execution function
    Downloads data from MAST and runs selected pipeline stages

Parameters (reduce function)
----------------------------
path : str
    Working directory for data processing (default: './')
target_short : str
    Short name for target (used in file naming)
target_name : str
    Full target name as appears in MAST
obs_id : str, optional
    Specific observation ID to download (if None, uses target_name)
proposal_id : str
    JWST proposal ID (e.g., '1584')
run_dl : bool
    Download data from MAST (requires MAST_API_TOKEN environment variable)
run_step1 : bool
    Run calwebb_detector1 (detector-level processing)
run_step2 : bool
    Run calwebb_spec2 (spectroscopic processing, creates _s3d.fits cubes)
run_step3 : bool
    Run calwebb_spec3 (combined products)

Usage Example
-------------
>>> from mirideep.reduce_script import reduce
>>> import os
>>> os.environ['MAST_API_TOKEN'] = 'your_token_here'
>>> reduce(target_short='mylup', target_name='MY-LUP', proposal_id='1584',
...        run_dl=True, run_step1=False, run_step2=True, run_step3=True)

Notes
-----
- Requires MAST_API_TOKEN environment variable for downloads
- Stage 2 processes each channel (1-4) separately with ifualign coordinate system
- Background exposures (BKGDTARG=True) are automatically removed
- Intermediate pipeline products are cleaned up to save disk space

Author
------
Klaus Pontoppidan (klaus.m.pontoppidan@jpl.nasa.gov)
"""

import os
import subprocess
import os.path as op
import numpy as np
import json

# The entire calwebb_detector1 pipeline
from jwst.pipeline import calwebb_detector1
from jwst.pipeline import calwebb_spec2
from jwst.pipeline import calwebb_spec3

from jwst import datamodels
from astroquery.mast import Observations
from astroquery.mast.missions import MastMissions
from astropy.io import fits


def reduce(path='./', target_short='wsb52', target_name='WSB-52', obs_id=None, proposal_id='1584',
           run_dl=True, run_step1=False, run_step2=True, run_step3=True):

    if run_dl:
        # Check for required MAST API token
        if 'MAST_API_TOKEN' not in os.environ:
            raise ValueError(
                "MAST_API_TOKEN environment variable must be set for downloading data.\n"
                "Get your token from https://auth.mast.stsci.edu/token and set it with:\n"
                "  export MAST_API_TOKEN='your_token_here'"
            )

        import glob
        for f in glob.glob("*rate.fits"):
            os.remove(f)

        my_session = Observations.login(token=os.environ['MAST_API_TOKEN'])

        if obs_id:
            missions = MastMissions(mission='jwst')
            obs = missions.query_criteria(program=proposal_id,observtn=obs_id,productLevel='2a',exp_type='MIR_MRS')
            products = missions.get_product_list(obs)
            condition = np.char.endswith(products['filename'],'rate.fits')
            rate_products = products[condition]
            missions.download_products(rate_products,flat=True)
        else:
            obs = Observations.query_criteria(obs_collection="JWST",proposal_id=proposal_id,target_name=[target_name])    
            products = Observations.get_product_list(obs)
            Observations.download_products(products,productSubGroupDescription='RATE',flat=True)

        #Clean out background exposures / the MAST API is so clunky that it is not possible to check for this before download
        for root, dirs, files in os.walk(path):
            for file in files:
                if '_rate.fits' in file:
                    hdr = fits.getheader(file)
                    # Use .get() to safely check for BKGDTARG key (may not exist in all FITS files)
                    if hdr.get('BKGDTARG', False):
                        os.remove(file)
                        print('Removing background exposure: ', file)


    import time
    time.sleep(10)

    if run_step1:
        for root, dirs, files in os.walk(path):
            for file in files:
                if '_uncal.fits' in file:
                    detector1 = calwebb_detector1.Detector1Pipeline()
                    detector1.call(os.path.join(root,file),output_dir='.',save_results=True)

    if run_step2:
        command = 'asn_from_list -o l2_asn.json -r DMSLevel2bBase *ifu*rate.fits'
        subprocess.run([command], shell=True)
        command = 'mv l2_asn.json '+target_short+'_spec2_asn.json'
        subprocess.run([command], shell=True)

        files = os.listdir(path)
        asnfiles = [file for file in files if '_spec2_' in file]
        for asnfile in asnfiles:
            f = open(asnfile)
            asn_data = json.load(f)
            f.close()
            for product in asn_data['products']:
                name = product['name']

                #have to do it in this clunky way because the channel mode doesn't work in spec2
                if 'short' in name:
                    parameter_dict = {"cube_build":{"channel":"1","output_file":name+'_ch1',"coord_system":'ifualign'},"photom":{"skip":False},
                                      "extract_1d":{"skip":True},"pixel_replace":{"skip":False,"algorithm":"mingrad"}}
                    spec2 = calwebb_spec2.Spec2Pipeline()
                    spec2.call(product['members'][0]['expname'],save_results=True,steps=parameter_dict)

                    parameter_dict = {"cube_build":{"channel":"2","output_file":name+'_ch2',"coord_system":'ifualign'},"photom":{"skip":False},
                                      "extract_1d":{"skip":True},"pixel_replace":{"skip":False,"algorithm":"mingrad"}}
                    spec2 = calwebb_spec2.Spec2Pipeline()
                    spec2.call(product['members'][0]['expname'],save_results=True,steps=parameter_dict)

                if 'long' in name:

                    parameter_dict = {"cube_build":{"channel":"3","output_file":name+'_ch3',"coord_system":'ifualign'},"photom":{"skip":False},
                                      "extract_1d":{"skip":True},"pixel_replace":{"skip":False,"algorithm":"mingrad"}}
                    spec2 = calwebb_spec2.Spec2Pipeline()
                    spec2.call(product['members'][0]['expname'],save_results=True,steps=parameter_dict)

                    parameter_dict = {"cube_build":{"channel":"4","output_file":name+'_ch4',"coord_system":'ifualign'},"photom":{"skip":False},
                                      "extract_1d":{"skip":True},"pixel_replace":{"skip":False,"algorithm":"mingrad"}}
                    spec2 = calwebb_spec2.Spec2Pipeline()
                    spec2.call(product['members'][0]['expname'],save_results=True,steps=parameter_dict)

    #Purging unneeded output files
    import glob
    for f in glob.glob("*mirifushort_s3d.fits"):
        os.remove(f)
    for f in glob.glob("*mirifulong_s3d.fits"):
        os.remove(f)

    if run_step3:
        command = 'asn_from_list -o '+target_short+'_l3_asn.json --product-name '+target_short+' *_cal.fits'
        subprocess.run([command], shell=True)
        spec3 = calwebb_spec3.Spec3Pipeline()
        parameter_dict = {"cube_build":{"output_type":"channel"},"pixel_replace":{"skip":False,"algorithm":"mingrad"},"adaptive_trace_model":{"skip":False}}
        spec3.call(target_short+'_l3_asn.json',output_dir='.',save_results=True,steps=parameter_dict)
