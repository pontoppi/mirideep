# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mirideep is a Python package for calibrating high signal-to-noise JWST MIRI MRS (Mid-Infrared Medium Resolution Spectrometer) data. The package performs advanced spectral extraction, fringe removal, and background subtraction beyond the standard JWST pipeline processing.

**Author:** Klaus Pontoppidan (klaus.m.pontoppidan@jpl.nasa.gov)  
**Current Version:** 9.5

## Installation & Setup

Install the package in development mode:
```bash
pip install -e .
```

The package includes calibration data (RSRFs - Relative Spectral Response Functions) stored in `mirideep/rsrfs/` which are bundled with the package.

## Architecture

### Core Components

**mirideep/core.py** (~850 lines)
- Contains `MiriDeepSpec` class - the primary interface for spectral extraction
- Key methods:
  - `run_extract()`: Main extraction pipeline that processes all channels/bands
  - `extract()`: Performs aperture photometry on individual cubes
  - `bg()`: Background estimation using nod or annulus methods
  - `shift_rsrf()`: Cross-correlation to align RSRFs with observed fringes
  - `scale()`: Inter-segment flux scaling to stitch spectral segments
  - `writespec()`: Outputs final 1D spectrum as FITS table

**mirideep/reduce_script.py** (~127 lines)
- `reduce()` function: Downloads data from MAST and runs JWST pipeline stages
- Executes calwebb_detector1, calwebb_spec2, and calwebb_spec3 pipelines
- Creates association files and processes data by channel
- Requires MAST_API_TOKEN environment variable

**mirideep/utils.py** (~50 lines)
- `fit_wavecorr()`: Wavelength calibration corrections using reference data
- Polynomial fitting to correct wavelength offsets

**mirideep/rsrfs/** 
- Numpy archives (.npz) and FITS files containing reference RSRF data for different calibration sources
- Multiple versions (5.0, 6.0, 6.1, 6.3, 7.1, 8.0, 8.1, 8.2, 8.3, 8.4, 9.5)
- CSV/DAT files for wavelength calibration and emissivity tables

**mirideep/examples/**
- `run.py`: Example extraction script for mylup observation (program 1584)
- `reduce_all.py`: Batch processing script that traverses multiple observation folders and extracts spectra from each

### Data Flow

1. Raw JWST data → JWST pipeline (stages 1-3) → _s3d.fits cubes
2. `find_cubes()`: Discovers all _s3d.fits files and organizes by channel/band/dither
3. For each channel-band setting:
   - `bg()`: Estimates background (nod subtraction or annulus)
   - `extract()`: Aperture photometry at each wavelength
   - `shift_rsrf()`: Aligns RSRF to observed fringe pattern
   - Defringe: spec1d / rsrf × standard_model
4. `scale()`: Stitches spectral segments across overlapping regions
5. `writespec()`: Outputs wavelength, flux, uncertainty, background

### Background Estimation Modes

**bg_types parameter** (set per channel in `__init__`):
- `'nod'`: Classic nod subtraction - median of other dithers (default for low-background)
- `'annulus'`: Spatial annulus around source - for high-background extended sources

### RSRF (Fringe Removal)

RSRFs are pre-computed from calibration star observations and stored as .npz files. The package:
1. Loads appropriate RSRF based on channel/band/dither
2. Uses cross-correlation (`shift_rsrf`) to align RSRF with observed spectrum
3. Divides observed spectrum by RSRF and multiplies by standard star model

### Calibration Sources

- Default: 'jena2' for ch2-4, 'hd163466_COM' for ch1
- RSRFs computed from observations of these calibrators
- `standard_model()` provides reference spectral shape (blackbody + emissivity)

## Key Parameters

**MiriDeepSpec initialization:**
- `bg_types`: Dict specifying background method per channel (e.g., `{'ch1':'nod','ch2':'nod','ch3':'annulus','ch4':'annulus'}`)
- `rrs`: Dict of aperture radii in units of diffraction limit (default 1.4)
- `standard`: Calibration source for ch2-4 (default 'jena2'). Can be a string or list of strings for multiple calibrators
- `ch1_standard`: Calibration source for ch1 (default 'hd163466_COM'). Can be a string or list of strings for multiple calibrators
- `wave_correct`: Apply wavelength corrections (default True)
- `shift_optimize`: Optimize RSRF shift via cross-correlation (default True)
- `single_shift`: Use median shift for all dithers vs individual shifts (default True)
- `mask_ratio`: Ratio threshold for masking bad pixels (default 20)
- `centroid_type`: Method for centroiding ('1dg' for 1D Gaussian)
- `source_cen`: Use source position instead of auto-centroiding (provide (x,y) tuple)
- `scale_to_segment`: Renormalize scales to a specific segment index (default False)

**Multiple Calibrators:**
When `standard` and/or `ch1_standard` are provided as lists, `run_extract()` will:
1. Extract the spectrum using each calibrator independently
2. Average the resulting flux arrays
3. Combine uncertainties in quadrature divided by sqrt(N)
4. Average the background estimates

This reduces systematic uncertainties from calibrator choice.

## Development Notes

### Version History

Recent changes focus on background subtraction improvements:
- v9.5: Fixed bug recording background scale error as spectral scale error
- v9.1-9.5: Extensive work on high-background subtraction and annulus-based estimation
- v8.x: Updated RSRFs with new pixel replacement algorithm
- Earlier versions: Core extraction and fringe removal pipeline

### Testing

No automated test suite is currently present. Testing is done via:
- Processing known calibration sources and comparing to expected results
- Visual inspection of fringe removal quality (use `plot_fringematch=True`)
- Checking centroid accuracy (use `plot_centroid=True`)

### Dependencies

Key external packages:
- JWST pipeline (`jwst` package): calwebb_detector1, calwebb_spec2, calwebb_spec3
- Astropy: FITS I/O, WCS, photometry, convolution, units, modeling
- Photutils: Aperture photometry and centroids
- Scipy: Signal processing (Savitzky-Golay, correlation, median filter, peak finding)
- Astroquery: MAST data downloads

### Common Workflows

The typical workflow is a two-step process: (1) download and reduce data through JWST pipeline, (2) extract high S/N spectrum with mirideep.

**Step 1: Download and reduce data (Level 2 & 3 processing)**

Uses `reduce_script.py` to download from MAST and run JWST pipeline stages 2-3:

```python
from mirideep.reduce_script import reduce

# Download and run pipeline (example: program 1584, observation mylup)
reduce(path='./', target_short='mylup', target_name='MY-LUP', 
       proposal_id='1584', run_dl=True, run_step1=False, run_step2=True, run_step3=True)
```

This produces `*_s3d.fits` cubes in the current directory.

**Step 2: Extract spectrum with mirideep**

Run from within the observation folder (e.g., `data_mylup/`) containing the `*_s3d.fits` files:

```python
from mirideep.core import MiriDeepSpec

# Standard extraction (example from mirideep/examples/run.py)
md = MiriDeepSpec(source='mylup', save_intermediate=True, standard='jena2',
                  rrs={'ch1':1.4,'ch2':1.3,'ch3':1.2,'ch4':1.1},
                  bg_types={'ch1':'nod','ch2':'nod','ch3':'nod','ch4':'nod'},
                  wave_correct=True, ch1_standard='hd163466_0723')
md.run_extract()
# Output: mylup_1d_v9.5.fits

# High-background source with annulus subtraction
md = MiriDeepSpec(source='my_source', 
                  bg_types={'ch1':'nod','ch2':'nod','ch3':'annulus','ch4':'annulus'})
md.run_extract()

# Multiple calibrators - averages results to reduce systematic uncertainties
md = MiriDeepSpec(source='mylup',
                  standard=['jena2', 'athalia2'],  # Extract with both, then average
                  ch1_standard=['hd163466_0723', 'hd163466_0823'],
                  rrs={'ch1':1.4,'ch2':1.3,'ch3':1.2,'ch4':1.1},
                  bg_types={'ch1':'nod','ch2':'nod','ch3':'nod','ch4':'nod'})
md.run_extract()
```

**Batch processing multiple observations**

See `mirideep/examples/reduce_all.py` for a convenience script that traverses multiple observation folders and extracts spectra from each one. The script:
1. Iterates through a list of observations from program 1584
2. Changes to each data directory (e.g., `data_mylup/`, `data_as209/`)
3. Runs `reduce()` for pipeline processing
4. Logs success/failure for each target

**Create custom RSRF from calibrator:**
```python
md = MiriDeepSpec(source='calibrator_name')
md.create_rsrf(standard='jena2', bg_type='nod')
```

### File Naming Conventions

- Input: `*_s3d.fits` - 3D spectral cubes from JWST pipeline
- Output: `{source}_1d_v{version}.fits` - 1D extracted spectrum
- Intermediate: `{source}_intermediates_v{version}.npz` - if `save_intermediate=True`

### Code Style Notes

- Uses astropy conventions for units and FITS handling
- Numpy arrays for spectral data
- Matplotlib for diagnostic plots
- Extensive use of sigma-clipping for combining dithers
- Background subtraction critical for accuracy - method selection depends on science case
