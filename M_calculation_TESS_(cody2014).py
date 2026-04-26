"""

Python version 3.9 was used

Specific Packages/Modules: lightkurve, scipy.signal.savgol_filter, astropy.stats.sigma_clip

"""
# Cody2014_M_for_TW_Hya.py
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.signal import savgol_filter
from astropy.stats import sigma_clip

# -------------------------------
# PARAMETERS (edit if needed)
# -------------------------------
TARGET = TARGET_NAME
SMOOTH_DAYS = 6    # Cody et al. (2014) used 2 hr for optical (CoRoT)
OUTLIER_SIGMA = 5.0   # drop points with |residual| > 5 sigma (measured on residuals)
DECILE = 0.10         # top/bottom 10%
USE_SAVGOL = True     # smoothing choice: Savitzky-Golay (True) or running median (False)

# -------------------------------
# Download and prepare lightcurve
# -------------------------------
searchres = lk.search_lightcurve(TARGET, mission="TESS", author="SPOC")
if len(searchres) == 0:
    raise RuntimeError("No SPOC TESS light curve found for target; try other authors or coordinates.")
lcs = searchres.download_all()
print(lcs)
lc = lcs[0].remove_nans()   # stitch sectors so we handle long baseline

time = lc.time.value              # days
flux = lc.flux.value.copy()       # flux (not normalized here yet)

# dt = np.diff(time)
# gap_idx = np.where(dt > 0.1)[0]   # ~1 day orbit gap

# orbit_starts = np.concatenate([[0], gap_idx + 1])
# orbit_ends   = np.concatenate([gap_idx + 1, [len(time)]])

# mask = np.zeros(len(time), dtype=bool)

# for s, e in zip(orbit_starts, orbit_ends):
#     if s + 100< e:
#         mask[s+100:e] = True

# time = time[mask]
# flux = flux[mask]

# # ensure positive flux (required for magnitude conversion). normalize to median:
# # flux_med = np.nanmedian(flux)
# # flux_rel = flux / flux_med       # relative flux (median ~1.0)
# flux_rel = lc.normalize().flux.value
# # convert to magnitudes like Cody et al. used (d = mag residuals)
# # small-signal approximation: mag = -2.5 * log10(flux_rel)
# mag = -2.5 * np.log10(flux_rel)

# --- MIT TESS magnitude definition ---
# m = -2.5 log10(cts/s) + 20.44

ZP = 20.44  # TESS zeropoint

# Raw TESS flux (cts/s)
F = flux

# Convert to calibrated TESS magnitudes
mag_raw = (-2.5 * np.log10(F)) + ZP

# Normalize magnitudes (this is equivalent to flux normalization)
mag = mag_raw - np.nanmedian(mag_raw)

# -------------------------------
# Smooth on SMOOTH_DAYS and subtract trend
# -------------------------------

dt_days = np.nanmedian(np.diff(time))  # cadence in days

if dt_days <= 0:
    raise RuntimeError("Time sampling error")

# Convert physical timescale → window length
window_length = int(SMOOTH_DAYS / dt_days)

# SavGol requirements
if window_length % 2 == 0:
    window_length += 1
window_length = max(5, window_length)   # avoid tiny unstable windows

if USE_SAVGOL:
    polyorder = 3
    smooth_mag = savgol_filter(mag, window_length, polyorder, mode='mirror')
else:
    from scipy.ndimage import median_filter
    smooth_mag = median_filter(mag, size=window_length, mode='nearest')



# residual light curve in magnitudes (this is "d(t)" in the paper)
d = mag - smooth_mag

# -------------------------------
# Identify and remove 5-sigma outliers (on the residuals)
# -------------------------------
# Use a robust sigma estimate (std of d) or astropy sigma_clip for robustness
# First compute sigma estimate
sigma0 = np.nanstd(d)
outlier_mask = np.abs(d) > OUTLIER_SIGMA * sigma0

# Filtered residuals (remove outliers)
d_f = d[~outlier_mask]

# If too many points removed, warn
if len(d_f) < 0.5 * len(d):
    print("Warning: more than half the points were flagged as outliers. Check smoothing/outlier thresholds.")

# -------------------------------
# Compute decile (top and bottom 10%) mean on outlier-filtered residuals
# -------------------------------
N = len(d_f)
n10 = max(1, int(np.floor(DECILE * N)))

d_sorted = np.sort(d_f)

bottom10 = d_sorted[:n10]
top10 = d_sorted[-n10:]
d10_mean = np.mean(np.concatenate([bottom10, top10]))

# -------------------------------
# Compute d_med and sigma_d (RMS) on outlier-filtered residuals
# -------------------------------
d_med = np.median(d_f)
sigma_d = np.std(d_f, ddof=0)   # population std (paper uses RMS of the light curve)

# -------------------------------
# Compute M
# -------------------------------
M = (d10_mean - d_med) / sigma_d

# -------------------------------
# Print and plot diagnostics
# -------------------------------
print("Cody2014-style M (magnitudes, 2-hr smoothing):", M)
print("d10_mean:", d10_mean, "d_med:", d_med, "sigma_d:", sigma_d)

# Diagnostic plots: raw mag, smoothed trend, residual histogram with deciles marked
plt.figure(figsize=(10,4))
plt.plot(time, flux, '.', ms = 2, alpha = 0.6, label = "Raw TESS Flux")
plt.xlabel('Time (days)')
plt.ylabel('TESS Flux (e/s)')
plt.legend()
plt.title(f'{TARGET}:Raw Flux')
plt.show()

plt.figure(figsize=(10,4))
plt.plot(time, mag_raw, '.', ms = 2, alpha = 0.6, label = "Raw TESS Mag")
plt.xlabel('Time (days)')
plt.ylabel('TESS Mag')
plt.legend()
plt.title(f'{TARGET}:Raw Mag')
plt.show()

plt.figure(figsize=(10,4))
plt.plot(time, mag, '.', ms=2, alpha=0.5, label='mag (raw)')
plt.plot(time, smooth_mag, '-', lw=1.5, color='C1', label=f'smoothed ({SMOOTH_HOURS} hr)')
plt.xlabel('Time (days)')
plt.ylabel('Relative magnitude (mag)')
plt.legend()
plt.title(f'{TARGET}: Normalised mag and smoothed trend; computed M = {M:.3f}')
plt.show()

plt.figure(figsize=(10,4))
plt.plot(time, d, '.', ms=2, alpha=0.5, label='mag (residual)')
# plt.plot(time, smooth_mag, '-', lw=1.5, color='C1', label=f'smoothed ({SMOOTH_HOURS} hr)')
plt.xlabel('Time (days)')
plt.ylabel('Relative magnitude (mag)')
plt.legend()
plt.title(f'{TARGET}: Residual LC; computed M = {M:.3f}')
plt.show()

# Residual histogram
plt.figure(figsize=(6,4))
plt.hist(np.asarray(d_f), bins='fd', edgecolor='k', alpha=0.7)
plt.axvline(np.percentile(d_f, 10), color='gray', linestyle='--', label='10th percentile')
plt.axvline(np.percentile(d_f, 90), color='gray', linestyle='--', label='90th percentile')
plt.axvline(d_med, color='red', label='median (filtered)')
plt.axvline(d10_mean, color='magenta', label='mean of top+bottom deciles')
plt.legend()
plt.xlabel('Residual magnitude d (mag)')
plt.ylabel('Count')
plt.title('Residual histogram (outlier-filtered)')
plt.show()
