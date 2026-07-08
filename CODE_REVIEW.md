# Code Review - Potential Issues and Recommendations

## Critical Issues

### 1. Bare `except:` Clauses (core.py)

**Location:** core.py:557, core.py:797

**Issue:** Using bare `except:` catches all exceptions including `KeyboardInterrupt` and `SystemExit`, which can make debugging difficult and hide serious errors.

**Lines 554-558:**
```python
if np.any(bsubs):
    try:
        spec1d[bsubs.flatten()] = np.interp(wave[bsubs].flatten(),wave[gsubs].flatten(),spec1d[gsubs].flatten())
    except:
        spec1d[bsubs.flatten()] = np.nan
```

**Lines 794-799:**
```python
try:
    fit = fitter_gauss(model_total, np.arange(maxlag*2), peakspec, maxiter=1000)
    lag_fit = fit.mean_0.value - maxlag + 1
except:
    print('cross correlation failed - no valid values. Assuming lag==0')
    lag_fit = 0
```

**Recommendation:** Replace with specific exception types:
```python
except (ValueError, IndexError) as e:
    # handle specific errors
```

### 2. Empty `lags` List Handling (core.py:213)

**Location:** core.py:213

**Issue:** If no dithers are found for a channel/band, `lags` will be empty and `np.median(lags)` will fail.

```python
lags.append(lag)

#Find the best median lag per module
lag_med = np.median(lags)
print(lags, lag_med)
```

**Recommendation:** Add check before computing median:
```python
if lags:
    lag_med = np.median(lags)
else:
    lag_med = 0
    print(f"Warning: No lags computed for {setting}, using lag_med=0")
```

### 3. Missing Error Handling for MAST_API_TOKEN (reduce_script.py:87) ✅ RESOLVED

**Location:** reduce_script.py:87

**Status:** Implemented comprehensive error handling with helpful instructions

**Implementation:**
```python
if 'MAST_API_TOKEN' not in os.environ:
    raise ValueError(
        "MAST_API_TOKEN environment variable must be set for downloading data.\n"
        "Get your token from https://auth.mast.stsci.edu/token and set it with:\n"
        "  export MAST_API_TOKEN='your_token_here'"
    )
my_session = Observations.login(token=os.environ['MAST_API_TOKEN'])
```

The error message now provides clear instructions including where to get the token and how to set it.

## Moderate Issues

### 4. Hardcoded `breakpoint()` Calls (core.py:636, 696)

**Location:** core.py:636, core.py:696

**Issue:** Left-in debugging breakpoints will halt execution in production.

```python
else:
    print('The annulus background option only works with forced photometry with a user-designated source position')
    breakpoint()
```

**Recommendation:** Replace with proper exceptions:
```python
else:
    raise ValueError('The annulus background option requires source_cen parameter to be set')
```

### 5. Inconsistent Use of `np.median` vs `np.nanmedian` (core.py:573)

**Location:** core.py:573

**Issue:** Mixing `np.nanmedian` and `np.median` in same calculation could cause issues if NaNs are present.

```python
scale = np.nanmedian(spec1ds[ii-1][osubs_left])/np.median(spec1ds[ii][osubs_right])
```

**Recommendation:** Use `np.nanmedian` consistently:
```python
scale = np.nanmedian(spec1ds[ii-1][osubs_left])/np.nanmedian(spec1ds[ii][osubs_right])
```

### 6. Hardcoded `time.sleep(10)` (reduce_script.py:112)

**Location:** reduce_script.py:112

**Issue:** Unclear why 10-second sleep is needed; may be workaround for race condition.

```python
import time
time.sleep(10)
```

**Recommendation:** Add comment explaining why sleep is necessary, or remove if no longer needed.

### 7. File Operations Without Error Handling (reduce_script.py:84-85)

**Location:** reduce_script.py:84-85

**Issue:** `os.remove()` called without checking if files exist or handling permissions errors.

```python
for f in glob.glob("*rate.fits"):
    os.remove(f)
```

**Recommendation:**
```python
for f in glob.glob("*rate.fits"):
    try:
        os.remove(f)
    except OSError as e:
        print(f"Warning: Could not remove {f}: {e}")
```

### 8. Potential KeyError When Accessing FITS Headers (core.py:106, reduce_script.py:106)

**Location:** reduce_script.py:106

**Issue:** Accessing `hdr['BKGDTARG']` without checking if key exists will raise `KeyError` for malformed FITS files.

```python
if hdr['BKGDTARG']:
    os.remove(file)
```

**Recommendation:**
```python
if hdr.get('BKGDTARG', False):
    os.remove(file)
```

## Minor Issues

### 9. Magic Number for Wavelength Threshold (core.py:801)

**Location:** core.py:801

**Issue:** Hardcoded wavelength threshold `40` without explanation.

```python
if np.mean(wave)>40:
    fig = plt.figure(figsize=(20,9))
```

**Recommendation:** Use named constant:
```python
DIAGNOSTIC_PLOT_WAVELENGTH_THRESHOLD = 40  # microns, only plot diagnostics for long wavelengths
if np.mean(wave) > DIAGNOSTIC_PLOT_WAVELENGTH_THRESHOLD:
```

### 10. Commented-Out Code (core.py, reduce_script.py)

**Location:** Multiple locations

**Issue:** Large blocks of commented-out code make the codebase harder to maintain.

Examples:
- core.py lines 116-150 (commented background subtraction algorithm)
- core.py lines 286-291 (commented silicate feature code)
- reduce_script.py lines 20-30 (commented logging setup)

**Recommendation:** Remove commented code and rely on git history, or move to separate experimental branch.

### 11. Inconsistent Error Messages

**Location:** core.py:635-636, 695-696

**Issue:** Error messages use different styles (print + breakpoint vs proper exceptions).

**Recommendation:** Use consistent exception handling throughout:
```python
raise ValueError("Error description") 
```

## Best Practice Recommendations

### 12. Add Input Validation ✅ RESOLVED

**Status:** Implemented comprehensive input validation in `MiriDeepSpec.__init__()`

**Added validations:**
- ✅ `bg_types` keys are valid channels ('ch1'-'ch4')
- ✅ `bg_types` values are valid methods ('nod', 'annulus', 'fit')
- ✅ `rrs` keys match `bg_types` keys
- ✅ Aperture radii are positive numbers
- ✅ `mask_ratio` is a positive number
- ✅ `source_cen` is False or a tuple/list of 2 numeric coordinates
- ✅ `scale_to_segment` is False or a non-negative integer

**Implementation:** The `__init__()` method now validates all input parameters and raises informative `ValueError` or `TypeError` exceptions with clear error messages when invalid inputs are provided.

### 13. Add Logging

Replace `print()` statements with proper logging using Python's `logging` module. The commented-out logger in reduce_script.py suggests this was intended.

### 14. Type Hints

Add type hints to function signatures for better code documentation and IDE support:
```python
def reduce(path: str = './', target_short: str = 'wsb52', 
           target_name: str = 'WSB-52', obs_id: Optional[str] = None,
           proposal_id: str = '1584', run_dl: bool = True,
           run_step1: bool = False, run_step2: bool = True,
           run_step3: bool = True) -> None:
```

### 15. Docstrings for Methods

Add docstrings to all methods in `MiriDeepSpec` class. Currently only the class has a docstring, but individual methods lack documentation.

## Summary

**High Priority:**
1. Fix bare except clauses (lines 557, 797)
2. Remove `breakpoint()` calls (lines 636, 696)
3. Add error handling for missing `MAST_API_TOKEN`
4. Handle empty `lags` list

**Medium Priority:**
5. Consistent use of `np.nanmedian`
6. Better FITS header key access with `.get()`
7. File operation error handling

**Low Priority:**
8. Remove commented code
9. Add logging
10. Add type hints
11. Document magic numbers
