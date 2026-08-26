"""
mirideep.batch_reduce - Parallel batch processing for MIRI data reduction

This module provides parallel batch processing capabilities for running
the mirideep reduction pipeline across multiple observations. It handles:

- Loading configuration from YAML files
- Parallel execution of reduction pipeline with configurable process count
- Logging and error handling across multiple observations
- Cleanup of intermediate products
- Isolation of each observation in its own process to avoid race conditions

Key Functions
-------------
run_batch_reduction() : Main batch processing function
    Run reduction pipeline on multiple observations in parallel

load_config() : Load YAML configuration
    Parse YAML config file with observations and pipeline parameters

Parameters (YAML Configuration)
-------------------------------
proposal_id : str
    JWST proposal ID (e.g., '1584')
observations : list of dict
    Each dict contains: dir, target_short, target_name, obs_id
run_dl : bool
    Download data from MAST
run_step1 : bool
    Run calwebb_detector1
run_step2 : bool
    Run calwebb_spec2
run_step3 : bool
    Run calwebb_spec3
max_workers : int, optional
    Maximum parallel processes (default: 8)
only : list of str, optional
    Restrict processing to specific observations, matched against each
    observation's `dir` or `target_short` (default: all observations)

YAML Configuration Example
---------------------------
proposal_id: '1584'
max_workers: 8

# Pipeline steps
run_dl: true
run_step1: false
run_step2: true
run_step3: true

# Restrict a run to specific observations (matched by `dir` or
# `target_short`). Omit or leave empty to process all observations.
# only:
#   - data_mylup
#   - wsb52

# Observations to process
observations:
  - dir: data_mylup
    target_short: mylup
    target_name: MY-LUP
    obs_id: 5
  - dir: data_wsb52
    target_short: wsb52
    target_name: WSB-52
    obs_id: 3

Usage Example
-------------
Command line usage:
    python -m mirideep.batch_reduce config.yaml --max-workers 4
    python -m mirideep.batch_reduce config.yaml --only data_mylup wsb52

Python usage:
    >>> from mirideep.batch_reduce import run_batch_reduction
    >>> results = run_batch_reduction('config.yaml', max_workers=8)
    >>> print(f"Success: {sum(1 for s in results.values() if s == 'success')}")
    >>> # Reprocess only a subset of observations
    >>> results = run_batch_reduction('config.yaml', only=['data_mylup'])

Notes
-----
- Observations are processed in parallel using ProcessPoolExecutor
- Each observation runs in its own process with isolated working directory
- This avoids race conditions when multiple observations call os.chdir()
- Failed observations don't stop the batch - errors are logged
- Results summary is written to log file

Author
------
Klaus Pontoppidan (klaus.m.pontoppidan@jpl.nasa.gov)
"""

import os
import subprocess
import logging
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional


def setup_logger(log_file: str = 'mirideep.log') -> logging.Logger:
    """
    Set up the logger for batch processing.

    Args:
        log_file: Path to the log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('mirideep')
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


def process_observation_worker(args):
    """
    Worker function for process pool - unpacks arguments and calls process_observation_impl.

    This is a top-level function required for multiprocessing to work properly.
    """
    return process_observation_impl(*args)


def process_observation_impl(
    obs: Dict,
    proposal_id: str,
    base_dir: str,
    run_dl: bool,
    run_step1: bool,
    run_step2: bool,
    run_step3: bool
) -> tuple:
    """
    Process a single observation in its own directory.

    This function runs in a separate process, so os.chdir() is safe.

    Args:
        obs: Observation dictionary with keys: dir, target_short, target_name, obs_id
        proposal_id: JWST proposal ID
        base_dir: Base directory for the batch run
        run_dl: Whether to run download step
        run_step1: Whether to run step 1
        run_step2: Whether to run step 2
        run_step3: Whether to run step 3

    Returns:
        Tuple of (target_short, status)
    """
    # Import here to avoid issues with multiprocessing
    from mirideep.reduce_script import reduce

    obs_dir = os.path.join(base_dir, obs['dir'])

    # Check if directory exists
    if not os.path.exists(obs_dir):
        return obs['target_short'], 'failed_no_dir'

    # Each process has its own working directory, so chdir is safe
    os.chdir(obs_dir)

    try:
        reduce(
            path='./',
            target_short=obs['target_short'],
            target_name=obs['target_name'],
            proposal_id=proposal_id,
            run_dl=run_dl,
            run_step1=run_step1,
            run_step2=run_step2,
            run_step3=run_step3,
            obs_id=obs.get('obs_id')
        )
        status = 'success'
    except Exception as e:
        status = f'failed: {str(e)}'

    # Cleanup
    try:
        subprocess.run(["rm *_crf.fits"], shell=True, check=False, cwd=obs_dir)
    except Exception:
        pass

    return obs['target_short'], status


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

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
    required_keys = ['proposal_id', 'observations']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required configuration keys: {missing_keys}")

    return config


def select_observations(observations: List[Dict], only: Optional[List[str]] = None) -> List[Dict]:
    """
    Filter observations down to a requested subset.

    Args:
        observations: Full list of observation dicts from the config
        only: List of identifiers to keep, matched against each observation's
            `dir` or `target_short` (case-sensitive). If None or empty,
            all observations are returned.

    Returns:
        Filtered list of observation dicts, in the original order

    Raises:
        ValueError: If an identifier in `only` matches no observation
    """
    if not only:
        return observations

    remaining = set(only)
    selected = []
    for obs in observations:
        if obs.get('dir') in remaining or obs.get('target_short') in remaining:
            selected.append(obs)
            remaining.discard(obs.get('dir'))
            remaining.discard(obs.get('target_short'))

    if remaining:
        raise ValueError(f"No observation matches: {sorted(remaining)}")

    return selected


def run_batch_reduction(
    config_path: str,
    max_workers: Optional[int] = None,
    log_file: str = 'mirideep.log',
    only: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Run batch reduction on multiple observations in parallel.

    Args:
        config_path: Path to YAML configuration file
        max_workers: Maximum number of parallel processes (overrides config, default: 8)
        log_file: Path to log file
        only: Restrict processing to specific observations, matched against
            each observation's `dir` or `target_short` (overrides the
            config's `only` list if given; default: all observations)

    Returns:
        Dictionary mapping target_short to status ('success', 'failed', etc.)

    Example:
        >>> results = run_batch_reduction('config.yaml', max_workers=4)
        >>> print(f"Success: {sum(1 for s in results.values() if s == 'success')}")
        >>> # Reprocess only a subset of observations
        >>> results = run_batch_reduction('config.yaml', only=['data_mylup'])
    """
    # Load configuration
    config = load_config(config_path)

    # Extract parameters
    proposal_id = config['proposal_id']
    if only is None:
        only = config.get('only')
    observations = select_observations(config['observations'], only)

    # Pipeline steps (default to False if not specified)
    run_dl = config.get('run_dl', False)
    run_step1 = config.get('run_step1', False)
    run_step2 = config.get('run_step2', False)
    run_step3 = config.get('run_step3', False)

    # Parallel processing settings
    if max_workers is None:
        max_workers = config.get('max_workers', 8)

    # Set up logger
    logger = setup_logger(log_file)

    # Store the base directory
    base_dir = os.getcwd()

    # Log configuration
    logger.info(f"Starting batch reduction for proposal {proposal_id}")
    if only:
        logger.info(f"Restricting to {len(observations)} observation(s): {only}")
    logger.info(f"Processing {len(observations)} observations with {max_workers} workers")
    logger.info(f"Pipeline steps - DL:{run_dl} Step1:{run_step1} Step2:{run_step2} Step3:{run_step3}")

    # Run observations in parallel using ProcessPoolExecutor
    # Each process gets its own working directory, avoiding race conditions
    results = {}

    # Prepare arguments for each observation
    job_args = [
        (obs, proposal_id, base_dir, run_dl, run_step1, run_step2, run_step3)
        for obs in observations
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_obs = {
            executor.submit(process_observation_worker, args): observations[i]
            for i, args in enumerate(job_args)
        }

        # Process results as they complete
        for future in as_completed(future_to_obs):
            obs = future_to_obs[future]
            try:
                target_short, status = future.result()
                results[target_short] = status
                logger.info(f"Completed {target_short} with status: {status}")
            except Exception as e:
                logger.error(f"Unexpected error processing {obs['target_short']}: {str(e)}")
                results[obs['target_short']] = 'error'

    # Return to base directory
    os.chdir(base_dir)

    # Summary
    success_count = sum(1 for status in results.values() if status == 'success')
    failed_count = len(results) - success_count

    logger.info(f"Batch reduction complete: {success_count} successful, {failed_count} failed")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run JWST MIRI batch reduction pipeline in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML configuration:
    proposal_id: '1584'
    max_workers: 8
    run_dl: true
    run_step1: false
    run_step2: true
    run_step3: true
    observations:
      - dir: data_mylup
        target_short: mylup
        target_name: MY-LUP
        obs_id: 5

Reprocess only specific observations:
    python -m mirideep.batch_reduce config.yaml --only data_mylup
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
        help='Maximum number of parallel processes (overrides config file, default: 8)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='mirideep.log',
        help='Path to log file (default: mirideep.log)'
    )
    parser.add_argument(
        '--only',
        type=str,
        nargs='+',
        default=None,
        metavar='ID',
        help='Restrict processing to specific observations, matched against '
             'each observation\'s dir or target_short (overrides the config '
             'file\'s only list; default: all observations)'
    )

    args = parser.parse_args()

    # Run batch reduction
    results = run_batch_reduction(
        config_path=args.config,
        max_workers=args.max_workers,
        log_file=args.log_file,
        only=args.only
    )

    # Print summary
    print(f"\nBatch reduction complete:")
    print(f"  Success: {sum(1 for s in results.values() if s == 'success')}")
    print(f"  Failed: {sum(1 for s in results.values() if s != 'success')}")
