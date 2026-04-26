import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d, median_filter
from astropy.timeseries import LombScargle

# -------------------------------
# USER OPTIONS
# -------------------------------
USE_SAVGOL = True

TARGET = TARGET_NAME
ZP = 20.44
SMOOTH_DAYS = 6
PERIOD_MIN, PERIOD_MAX = 0.2, 20.0

# -------------------------------
# Load light curve
# -------------------------------
lcs = lk.search_lightcurve(
    TARGET, mission="TESS", author="SPOC"
).download_all()

lc = lcs[0].remove_nans()

time = lc.time.value
flux = lc.flux.value

# -------------------------------
# Convert to magnitudes
# -------------------------------
mask = flux > 0
time = time[mask]
flux = flux[mask]

mag_raw = -2.5 * np.log10(flux) + ZP
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
d = mag - smooth_mag   # Cody d(t)

# -------------------------------
# ACF: candidate period
# -------------------------------
acf = correlate(d, d, mode="full")
acf = acf[len(acf)//2:]
acf /= acf[0]

dt = np.median(np.diff(time))
lags = np.arange(len(acf)) * dt

peaks, _ = find_peaks(acf, height=0.15, distance=20)
candidate_periods = lags[peaks]
candidate_periods = candidate_periods[
    (candidate_periods > PERIOD_MIN) &
    (candidate_periods < PERIOD_MAX)
]

if len(candidate_periods) == 0:
    raise RuntimeError("No significant ACF peaks found")

P_acf = candidate_periods[0]

# -------------------------------
# Lomb–Scargle confirmation (+- 15% in frequency)
# -------------------------------
ls = LombScargle(time, d)

# --- Wide LS for diagnostics only ---
periods_plot = np.linspace(0.1, 15.0, 5000)
frequencies_plot = 1.0 / periods_plot
power_plot = ls.power(frequencies_plot)

# --- LS confirmation window (USED FOR PERIOD SELECTION) ---
f_acf = 1.0 / P_acf
fmin = 0.85 * f_acf
fmax = 1.15 * f_acf

frequencies_zoom = np.linspace(fmin, fmax, 2000)
power_zoom = ls.power(frequencies_zoom)

best_freq = frequencies_zoom[np.argmax(power_zoom)]
best_period = 1.0 / best_freq

print(f"ACF period = {P_acf:.3f} d")
print(f"LS-confirmed period = {best_period:.3f} d")

# -------------------------------
# Phase folding
# -------------------------------
t0 = np.min(time)
phase = ((time - t0) % best_period) / best_period

idx = np.argsort(phase)
phase_s = phase[idx]
d_s = d[idx]

# -------------------------------
# Smoothed phase curve (25% boxcar)
# -------------------------------
phase_width = 0.25   # Cody et al.
window = int(phase_width * len(d_s))
window = max(3, window | 1)

template_s = uniform_filter1d(
    d_s, size=window, mode="wrap"
)

template = np.zeros_like(d)
template[idx] = template_s

# -------------------------------
# Residuals + Q
# -------------------------------
residuals = d - template
Q = np.var(residuals) / np.var(d)

print(f"Cody Q = {Q:.3f}")

# ======================================================
# ================= DIAGNOSTICS ========================
# ======================================================

# Raw LC Flux
plt.figure(figsize=(8,4))
plt.plot(time, flux, lw=1, color="black")
plt.xlabel("Time (days)")
plt.ylabel("TESS Flux (e/s)")
plt.title(f"TESS Flux {TARGET}")
plt.show()

# Raw LC Mag
plt.figure(figsize=(8,4))
plt.plot(time, mag_raw, lw=1, color="black")
plt.xlabel("Time (days)")
plt.ylabel("TESS Magnitude")
plt.title(f"TESS Magnitude {TARGET}")
plt.show()

# Normalised Mag
plt.figure(figsize=(10,4))
plt.plot(time, mag, '.', ms=2, alpha=0.2, color = 'gray', label='mag (raw)')
plt.plot(time, smooth_mag, '-', lw=1.5, color='C1', label=f'smoothed ({SMOOTH_HOURS} hr)')
plt.plot(time, d, 'o', ms=1, alpha=0.5, color = 'purple', label='mag (residual)')
plt.xlabel('Time (days)')
plt.ylabel(f'Relative magnitude (mag)')
plt.legend()
plt.title(f'{TARGET}: Normalised mag and smoothed trend; computed M = {M:.3f}')
plt.show()

# ACF
plt.figure(figsize=(8,4))
plt.plot(lags, acf, lw=1)
plt.axvline(P_acf, color='red', ls='--')
plt.xlabel("Lag (days)")
plt.ylabel("ACF")
plt.title(f"ACF {TARGET}")
plt.xlim(0, 15)
plt.show()

# LS (restricted)
plt.figure(figsize=(8,4))
plt.plot(periods_plot, power_plot, color='black', lw=1)
plt.axvline(best_period, color='red', ls='--',
            label=f'Selected period = {best_period:.2f} d')
plt.xlabel("Period (days)")
plt.ylabel("Lomb–Scargle Power")
plt.title(f"Lomb–Scargle Periodogram {TARGET}")
plt.xlim(0, 15)
plt.legend()
plt.show()

# Phase-folded
plt.figure(figsize=(6,4))
plt.scatter(phase_s, d_s, s=3, alpha=0.3)
plt.plot(phase_s, template_s, color='red', lw=2)
plt.xlabel("Phase")
plt.ylabel("Residual magnitude")
plt.title(f"Phase-folded LC + 25% boxcar {TARGET}")
plt.show()

# Periodic model over raw
plt.figure(figsize=(10,4))
plt.plot(time, d, lw=0.4, alpha=0.6, label="d(t)")
plt.plot(time, template, lw=2, color='red', label="Periodic model")
plt.legend()
plt.xlabel("Time (days)")
plt.ylabel("Residual magnitude")
plt.title(f"Periodic template over data {TARGET}")
plt.show()

# Final residuals
plt.figure(figsize=(10,3))
plt.plot(time, residuals, lw=0.5)
plt.xlabel("Time (days)")
plt.ylabel("Residual magnitude")
plt.title(f"Final residuals {TARGET}")
plt.show()

###########