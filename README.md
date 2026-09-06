# mirideep

A Python package for calibrating high signal-to-noise JWST MIRI MRS (Mid-Infrared Medium Resolution Spectrometer) data. The package performs advanced spectral extraction, fringe removal, and background subtraction beyond the standard JWST pipeline processing.

## Features

- **Advanced Background Subtraction**: Support for both nod subtraction and annulus-based methods for high-background extended sources
- **Fringe Removal**: RSRF-based defringing using pre-computed calibration data with cross-correlation optimization
- **Multi-Channel Processing**: Handles all MIRI MRS channels (1-4) and bands (short, medium, long)
- **Spectral Stitching**: Automated scaling and stitching of overlapping spectral segments
- **Pipeline Integration**: Direct integration with JWST calibration pipeline stages 2-3
- **Batch Processing**: Parallel batch processing with YAML configuration support

## Installation

### Requirements

- Python 3.x
- JWST pipeline (`jwst` package)
- Astropy
- Photutils
- Scipy
- Numpy
- Matplotlib
- Astroquery (for MAST data downloads)
- PyYAML (for batch processing configuration)

### Install

Clone the repository and install in development mode:

```bash
git clone https://github.com/yourusername/mirideep.git
cd mirideep
pip install -e .
```

## Quick Start

### 1. Download and Reduce Data (Pipeline Stages 2-3)

First, set your MAST API token:
```bash
export MAST_API_TOKEN="your_token_here"
```

Then download and process observations:

```python
from mirideep.reduce_script import reduce

# Download data from MAST and run JWST pipeline
reduce(path='./', 
       target_short='mylup', 
       target_name='MY-LUP', 
       proposal_id='1584',
       run_dl=True,      # Download from MAST
       run_step1=False,  # Skip detector1 (use rate files)
       run_step2=True,   # Run spec2 pipeline
       run_step3=True)   # Run spec3 pipeline
```

This produces `*_s3d.fits` spectral cubes.

### 2. Extract High S/N Spectrum

Run from the directory containing the `*_s3d.fits` files:

```python
from mirideep.core import MiriDeepSpec

# Standard extraction with nod subtraction
md = MiriDeepSpec(source='mylup',
                  standard='jena2',
                  ch1_standard='hd163466_0723',
                  wave_correct=True)
md.run_extract()
```

Output: `mylup_1d_v9.6.fits` - a 1D spectrum in FITS table format with columns for wavelength, flux density, uncertainty, and background.

### 3. High-Background Sources

For extended sources with high background:

```python
md = MiriDeepSpec(source='my_source',
                  bg_types={'ch1':'nod', 'ch2':'nod', 'ch3':'nod', 'ch4':'nod'},
                  rrs={'ch1':1.4, 'ch2':1.3, 'ch3':1.2, 'ch4':1.1})
md.run_extract()
```

## Batch Processing (NEW)

### Batch Data Reduction

Process multiple observations in parallel with YAML configuration:

### Create a configuration file (`reduce_config.yaml`):

```yaml
proposal_id: '1584'
max_workers: 8  # Number of parallel threads

# Pipeline steps
run_dl: true
run_step1: false
run_step2: true
run_step3: true

# Observations
observations:
  - dir: data_mylup
    target_short: mylup
    target_name: MY-LUP
    obs_id: 5
  - dir: data_szcha
    target_short: szcha
    target_name: SZ-CHA
    obs_id: 3
```

### Run batch processing:

```bash
# Use default settings (8 workers)
python -m mirideep.batch_reduce reduce_config.yaml

# Override thread count
python -m mirideep.batch_reduce reduce_config.yaml --max-workers 4
```

### Or use Python API:

```python
from mirideep.batch_reduce import run_batch_reduction

results = run_batch_reduction('reduce_config.yaml', max_workers=8)
print(f"Success: {sum(1 for s in results.values() if s == 'success')}")
```

### Features:
- **Parallel execution** with configurable process count
- **YAML configuration** for easy editing
- **Real-time logging** of progress
- **Robust error handling** - failed observations don't stop the batch
- **Process isolation** - each observation runs independently

### Batch Spectral Extraction

Extract 1D spectra from multiple observations in parallel:

#### Create extraction config (`extract_config.yaml`):

```yaml
max_workers: 4

default_params:
  standard: jena2
  ch1_standard: hd163466_0723
  rrs:
    ch1: 1.4
    ch2: 1.3
    ch3: 1.2
    ch4: 1.1
  bg_types:
    ch1: nod
    ch2: nod
    ch3: nod
    ch4: nod
  wave_correct: true

observations:
  - dir: data_twhya
    source: twhya
  - dir: data_aatau
    source: aatau
    wave_correct: false  # Override default
```

#### Run batch extraction:

```bash
# Use default settings (4 workers)
python -m mirideep.batch_extract extract_config.yaml

# Override worker count
python -m mirideep.batch_extract extract_config.yaml --max-workers 2
```

#### Or use Python API:

```python
from mirideep.batch_extract import run_batch_extraction

results = run_batch_extraction('extract_config.yaml', max_workers=4)
print(f"Success: {sum(1 for s in results.values() if s == 'success')}")
```

## Examples

The `mirideep/examples/` directory contains practical examples:

- **`reduce_all.py`**: Downloads and reduces observations from program 1584 (mylup example)
- **`data_mylup/run.py`**: Extraction script for the mylup observation
- **`run_all.py`**: Batch processing script that traverses multiple data directories

To run the example:
```bash
cd mirideep/examples
python reduce_all.py  # Download and reduce
cd data_mylup
python run.py         # Extract spectrum
```

## Key Parameters

### Background Estimation Methods

- `'nod'`: Classic nod subtraction using median of other dithers (default, best for low-background point sources)
- `'annulus'`: Spatial annulus around source (for high-background extended sources)

### Aperture Radii

- `rrs`: Dictionary of aperture radii per channel in units of diffraction limit (default: 1.4 for all channels)

### Calibration Sources

- `standard`: Calibration source for channels 2-4 (default: 'jena2')
- `ch1_standard`: Calibration source for channel 1 (default: 'hd163466_0723')

See [CLAUDE.md](CLAUDE.md) for complete parameter documentation and architecture details.

## Output Format

The extracted 1D spectrum is saved as a FITS binary table with columns:

- `wavelength` (microns)
- `fluxdensity` (Jy)
- `fluxdensity_stddev` (Jy)
- `background` (MJy/sr)

## Version History

- **v9.6** (Current): Added intermediate diagnostic plotting (spectra and cross-correlation fits); moved hardcoded calibrator data to `rsrfs/calibrators.yaml`
- **v9.5**: Fixed bug recording background scale error as spectral scale error; updated RSRFs
- **v9.1-9.4**: Extensive work on high-background subtraction and annulus-based estimation
- **v8.x**: Updated RSRFs with new pixel replacement algorithm
- **v7.x and earlier**: Core extraction and fringe removal pipeline

## Documentation

- [CLAUDE.md](CLAUDE.md) - Detailed technical documentation and architecture
- See docstrings in `mirideep/core.py` for API reference

## Contributing

This package is actively developed for processing JWST MIRI MRS observations. Contributions and bug reports are welcome.

## Author

Klaus Pontoppidan (klaus.m.pontoppidan@jpl.nasa.gov)

## License

MIT License - see [setup.py](setup.py) for details

## Citation

If you use this package in your research, please cite:
- Pontoppidan, K. M., et al. 2024, ApJ, 963, 2 ([doi:10.3847/1538-4357/ad20f0](https://doi.org/10.3847/1538-4357/ad20f0))

## Acknowledgments

This work uses calibration data derived from JWST observations and benefits from the JWST calibration pipeline developed by STScI.
