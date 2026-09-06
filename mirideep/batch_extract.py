"""
mirideep.batch_extract - Parallel batch spectral extraction for MIRI data

This module provides parallel batch processing for running MiriDeepSpec spectral
extraction across multiple observations. It handles:

- Loading configuration from YAML files
- Parallel execution with configurable process count
- Per-observation extraction parameters (standards, apertures, background methods)
- Logging and error handling
- Process isolation to avoid race conditions

Key Functions
-------------
run_batch_extraction() : Main batch extraction function
    Run MiriDeepSpec extraction on multiple observations in parallel

load_extract_config() : Load YAML configuration
    Parse YAML config file with observation-specific extraction parameters

Parameters (YAML Configuration)
-------------------------------
max_workers : int, optional
    Maximum parallel processes (default: 4)
default_params : dict, optional
    Default extraction parameters applied to all observations
observations : list of dict
    Each dict contains:
        - dir: observation subdirectory (e.g., 'data_twhya')
        - source: source name (e.g., 'twhya')
        - standard: calibration standard or list of standards (e.g., 'jena2' or ['jena2', 'athalia3'])
        - ch1_standard: channel 1 standard or list (e.g., 'hd163466_0723' or ['hd163466_0723', 'hd163466_0823'])
        - rrs: aperture radii per channel (e.g., {'ch1':1.4, 'ch2':1.3, ...})
        - bg_types: background method per channel (e.g., {'ch1':'nod', ...})
        - wave_correct: apply wavelength correction (bool)
        - plot_centroid: plot centroid (bool)
        - source_cen: source coordinates (tuple, optional)
        - save_intermediate: save intermediate products (bool)

Note: If multiple standards are specified (as a list), MiriDeepSpec will internally
use all standards and average the results. This produces a single averaged output file.

YAML Configuration Example
---------------------------
# Default parameters applied to all observations (can be overridden per observation)
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
  plot_centroid: false
  save_intermediate: true

# Observations to extract
observations:
  - dir: data_twhya
    source: twhya
    # Uses defaults above

  - dir: data_aatau
    source: aatau
    wave_correct: false  # Override default

  - dir: data_hmlup
    source: hmlup
    bg_types:  # Override default
      ch1: nod
      ch2: nod
      ch3: annulus
      ch4: annulus
    source_cen: [165.4657821, -34.7048163]

  - dir: data_comparison
    source: comparison
    standard: [jena2, athalia3]  # Use multiple standards (MiriDeepSpec averages them)
    ch1_standard: [hd163466_0723, hd163466_0823]  # Multiple ch1 standards (averaged)
    # This creates a single averaged output file

Usage Example
-------------
Command line usage:
    python -m mirideep.batch_extract extract_config.yaml --max-workers 4

Python usage:
    >>> from mirideep.batch_extract import run_batch_extraction
    >>> results = run_batch_extraction('extract_config.yaml', max_workers=4)
    >>> print(f"Success: {sum(1 for s in results.values() if s == 'success')}")

Notes
-----
- Extractions are processed in parallel using ProcessPoolExecutor
- Each extraction runs in its own process with isolated working directory
- Failed extractions don't stop the batch - errors are logged
- Output 1D spectra are saved as *_1d_v9.6.fits in each observation directory

Author
------
Klaus Pontoppidan (klaus.m.pontoppidan@jpl.nasa.gov)
"""

import os
import logging
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional


def setup_logger(log_file: str = 'mirideep_extract.log') -> logging.Logger:
    """
    Set up the logger for batch extraction.

    Args:
        log_file: Path to the log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('mirideep_extract')
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create file handler
    fh = logging.FileHandler(log_file, 'a')
    fh.setFormatter(logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Also log to console
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s'))
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger


def extract_observation_worker(args):
    """
    Worker function for process pool - unpacks arguments and calls extraction.

    This is a top-level function required for multiprocessing to work properly.
    """
    return extract_observation_impl(*args)


def extract_observation_impl(
    obs: Dict,
    base_dir: str
) -> tuple:
    """
    Run MiriDeepSpec extraction for a single observation.

    This function runs in a separate process, so os.chdir() is safe.
    Supports multiple standards - will run extraction once per standard combination.

    Args:
        obs: Observation dictionary with extraction parameters
        base_dir: Base directory for the batch run

    Returns:
        Tuple of (source, status)
    """
    # Import here to avoid issues with multiprocessing
    from mirideep.core import MiriDeepSpec

    obs_dir = os.path.join(base_dir, obs['dir'])
    source = obs['source']

    # Check if directory exists
    if not os.path.exists(obs_dir):
        return source, 'failed_no_dir'

    # Each process has its own working directory, so chdir is safe
    os.chdir(obs_dir)

    # Get standards - support single values, lists, or comma-separated strings
    # MiriDeepSpec handles multiple standards internally (averages them)
    standard = obs.get('standard', 'jena2')
    if isinstance(standard, str) and ',' in standard:
        # Handle comma-separated strings: "jena2,athalia3" -> ["jena2", "athalia3"]
        standard = [s.strip() for s in standard.split(',')]
    # Otherwise keep as-is (string or list) - MiriDeepSpec handles both

    ch1_standard = obs.get('ch1_standard', 'hd163466_0723')
    if isinstance(ch1_standard, str) and ',' in ch1_standard:
        # Handle comma-separated strings
        ch1_standard = [s.strip() for s in ch1_standard.split(',')]
    # Otherwise keep as-is (string or list) - MiriDeepSpec handles both

    try:
        # Build MiriDeepSpec arguments
        mds_kwargs = {
            'source': source,
            'save_intermediate': obs.get('save_intermediate', True),
            'standard': standard,  # Can be string or list - MiriDeepSpec handles both
            'ch1_standard': ch1_standard,  # Can be string or list
            'rrs': obs.get('rrs', {'ch1': 1.4, 'ch2': 1.3, 'ch3': 1.2, 'ch4': 1.1}),
            'bg_types': obs.get('bg_types', {'ch1': 'nod', 'ch2': 'nod', 'ch3': 'nod', 'ch4': 'nod'}),
            'wave_correct': obs.get('wave_correct', True),
            'plot_centroid': obs.get('plot_centroid', False)
        }

        # Add optional source_cen if provided
        if 'source_cen' in obs:
            source_cen = obs['source_cen']
            if isinstance(source_cen, list) and len(source_cen) == 2:
                mds_kwargs['source_cen'] = tuple(source_cen)

        # Create MiriDeepSpec instance and run extraction
        # MiriDeepSpec will internally average if multiple standards provided
        mds = MiriDeepSpec(**mds_kwargs)
        mds.run_extract()

        status = 'success'
    except Exception as e:
        status = f'failed: {str(e)}'

    return source, status


def load_extract_config(config_path: str) -> Dict:
    """
    Load extraction configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        KeyError: If required keys are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required keys
    if 'observations' not in config:
        raise KeyError("Missing required configuration key: 'observations'")

    # Apply default_params to all observations
    if 'default_params' in config:
        defaults = config['default_params']
        for obs in config['observations']:
            # Apply defaults for missing keys
            for key, value in defaults.items():
                if key not in obs:
                    obs[key] = value

    return config


def run_batch_extraction(
    config_path: str,
    max_workers: Optional[int] = None,
    log_file: str = 'mirideep_extract.log'
) -> Dict[str, str]:
    """
    Run batch extraction on multiple observations in parallel.

    Args:
        config_path: Path to YAML configuration file
        max_workers: Maximum number of parallel processes (overrides config, default: 4)
        log_file: Path to log file

    Returns:
        Dictionary mapping source to status ('success', 'failed', etc.)

    Example:
        >>> results = run_batch_extraction('extract_config.yaml', max_workers=4)
        >>> print(f"Success: {sum(1 for s in results.values() if s == 'success')}")
    """
    # Load configuration
    config = load_extract_config(config_path)

    # Extract parameters
    observations = config['observations']

    # Parallel processing settings
    if max_workers is None:
        max_workers = config.get('max_workers', 4)

    # Set up logger
    logger = setup_logger(log_file)

    # Store the base directory
    base_dir = os.getcwd()

    # Log configuration
    logger.info(f"Starting batch extraction")
    logger.info(f"Processing {len(observations)} observations with {max_workers} workers")

    # Run extractions in parallel using ProcessPoolExecutor
    # Each process gets its own working directory, avoiding race conditions
    results = {}

    # Prepare arguments for each observation
    job_args = [(obs, base_dir) for obs in observations]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_obs = {
            executor.submit(extract_observation_worker, args): observations[i]
            for i, args in enumerate(job_args)
        }

        # Process results as they complete
        for future in as_completed(future_to_obs):
            obs = future_to_obs[future]
            try:
                source, status = future.result()
                results[source] = status
                logger.info(f"Completed {source} with status: {status}")
            except Exception as e:
                logger.error(f"Unexpected error processing {obs['source']}: {str(e)}")
                results[obs['source']] = 'error'

    # Return to base directory
    os.chdir(base_dir)

    # Summary
    success_count = sum(1 for status in results.values() if status == 'success')
    failed_count = len(results) - success_count

    logger.info(f"Batch extraction complete: {success_count} successful, {failed_count} failed")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run JWST MIRI batch spectral extraction in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML configuration:
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
        """
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel processes (overrides config file, default: 4)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='mirideep_extract.log',
        help='Path to log file (default: mirideep_extract.log)'
    )

    args = parser.parse_args()

    # Run batch extraction
    results = run_batch_extraction(
        config_path=args.config,
        max_workers=args.max_workers,
        log_file=args.log_file
    )

    # Print summary
    print(f"\nBatch extraction complete:")
    print(f"  Success: {sum(1 for s in results.values() if s == 'success')}")
    print(f"  Failed: {sum(1 for s in results.values() if s != 'success')}")
