import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy.pyTiming import pyPDM
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
%matplotlib qt

# ===============================
# GLOBAL STYLE
# ===============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11
})

def style_axes(ax):
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)
    ax.grid(alpha=0.3)

# ===============================
# 1. DOWNLOAD DATA
# ===============================
target_name = TARGET

search_result = lk.search_lightcurve(target_name, radius=0.5, mission="TESS", author="SPOC")
if len(search_result) == 0:
    raise ValueError("No TESS SPOC light curves found.")

lc_collection = search_result.download_all()

# ===============================
# 2. ORGANIZE SECTORS
# ===============================
sector_lcs = {}

for lc, row in zip(lc_collection, search_result.table):
    # clean_lc = lc.remove_nans().normalize()[500:]
    clean_lc = lc.remove_nans().normalize()
    sector = int(row['mission'].split()[-1])
    key = f"sec{sector}"
    sector_lcs[key] = clean_lc

# ===============================
# 3. SPLIT INTO ORBITS
# ===============================
orbit_duration = 5.7
sector_orbit_dict = {}

for sec_key, lc in sector_lcs.items():
    time = lc.time.value
    t0 = np.min(time)

    for i in range(2):
        start = t0 + 2 * orbit_duration * i
        end   = t0 + 2 * orbit_duration * (i + 1)

        seg = lc[(time >= start) & (time < end)]
        if len(seg) > 0:
            sector_orbit_dict[f"{sec_key}_o{i+1}"] = seg

# ===============================
# 4. AUTO LOOP OVER ALL CHOICES
# ===============================

all_choices = []

# Add sector-wise choices
all_choices.extend(sector_lcs.keys())

# Add orbit-wise choices
all_choices.extend(sector_orbit_dict.keys())

# Add global
all_choices.insert(0, "all")

print("Running analysis for:", all_choices)

# ===============================
# LOOP OVER EACH CHOICE
# ===============================
for choice in all_choices:

    print(f"\n===== Processing: {choice} =====")

    # -----------------------------
    # Select LC list
    # -----------------------------
    if choice == "all":
        lc_list = list(sector_orbit_dict.values())
    elif choice in sector_lcs:
        lc_list = [lc for key, lc in sector_orbit_dict.items() if key.startswith(choice)]
    elif choice in sector_orbit_dict:
        lc_list = [sector_orbit_dict[choice]]
    else:
        continue

    # ===============================
    # 5. SAVGOL FLATTENING
    # ===============================
    from lightkurve import LightCurveCollection
    SMOOTH_HOURS = 36
    # window_hours = 24
    lc_flat_list, trend_list = [], []

    for lc in lc_list:
        time = lc.time.value
        flux = lc.flux.value
        
        dt_days = np.nanmedian(np.diff(time))
        dt_hours = dt_days * 24
        
        window_points = int(round(SMOOTH_HOURS / dt_hours))
        
        # Ensure odd
        if window_points % 2 == 0:
            window_points += 1
        
        # Ensure safe minimum
        window_points = max(5, window_points)

        trend_flux = savgol_filter(flux, window_points, 3, mode = 'mirror')
        flat_flux = flux / trend_flux

        lc_trend = lk.LightCurve(time=lc.time, flux=trend_flux)
        lc_flat = lk.LightCurve(time=lc.time, flux=flat_flux)

        trend_list.append(lc_trend)
        lc_flat_list.append(lc_flat)

    if len(lc_flat_list) > 1:
        lc_flat = LightCurveCollection(lc_flat_list).stitch()
        trend = LightCurveCollection(trend_list).stitch()
    else:
        lc_flat = lc_flat_list[0]
        trend = trend_list[0]

    # ===============================
    # 6. LC PLOT
    # ===============================
    fig, ax = plt.subplots(figsize=(12,4))

    for i, lc in enumerate(lc_list):
        ax.plot(lc.time.value, lc.flux.value, alpha=0.4)

    ax.plot(trend.time.value, trend.flux.value, color='red', lw=1.2)
    ax.plot(lc_flat.time.value, lc_flat.flux.value, color='black', lw=1)

    ax.set_title(f"{target_name} — {choice} (SavGol {SMOOTH_HOURS} hr)")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Normalized Flux")

    style_axes(ax)
    plt.tight_layout()
    plt.show()

    # ===============================
    # 7. PDM
    # ===============================
    time = lc_flat.time.value
    flux = lc_flat.flux.value

    pdm = pyPDM.PyPDM(time, flux)
    scanner = pyPDM.Scanner(minVal=0.1, maxVal=20, dVal=0.01, mode="period")

    periods, thetas = pdm.pdmEquiBinCover(10, 3, scanner)
    best_period = periods[np.argmin(thetas)]

    # ===============================
    # 8. GAUSSIAN FIT
    # ===============================
    def neg_gaussian(x, a, x0, sigma, c):
        return c - a * np.exp(-((x - x0)**2) / (2 * sigma**2))

    window = 0.25 * best_period
    mask = (periods >= best_period - window) & (periods <= best_period + window)

    x_fit, y_fit = periods[mask], thetas[mask]

    p0 = [
        np.max(y_fit) - np.min(y_fit),
        best_period,
        window/4,
        np.max(y_fit)
    ]

    try:
        popt, _ = curve_fit(neg_gaussian, x_fit, y_fit, p0=p0, maxfev=20000)
        a_fit, x0_fit, sigma_fit, c_fit = popt
        hwhm = 1.177 * sigma_fit

        print(f"{choice}: {x0_fit:.3f} ± {hwhm:.3f} d")

    except:
        x0_fit, hwhm = best_period, np.nan

    # ===============================
    # 9. PHASE FOLD
    # ===============================
    t0 = np.min(time)
    phase = ((time - t0) % best_period) / best_period

    idx = np.argsort(phase)
    phase_s, flux_s = phase[idx], flux[idx]

    # ===============================
    # 10. PDM + PHASE PLOT
    # ===============================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    # --- PDM ---
    ax1.plot(periods, thetas, 'k-', lw=1)

    try:
        x_dense = np.linspace(x_fit.min(), x_fit.max(), 500)
        y_dense = neg_gaussian(x_dense, *popt)
        ax1.plot(x_dense, y_dense, 'r-', lw=1)
        ax1.axvspan(x0_fit - hwhm, x0_fit + hwhm, color='gray', alpha=0.2)
    except:
        pass

    # Detection thresholds
    f_levels = [0.25, 0.5]
    y_theta = [1 / (1 + f) for f in f_levels]

    for f_val, y_val in zip(f_levels, y_theta):
        color = 'orange' if f_val == 0.25 else 'red'
        ax1.axhline(y_val, color=color, linestyle='--', lw=1.5)
        ax1.text(8, y_val+0.02, f"f = {f_val}", ha = 'center', va = 'center', fontsize = 12)
        # --- Shaded detection zones ---
        ax1.axhspan(y_theta[1], y_theta[0], color='orange', alpha=0.1)  # moderate
        ax1.axhspan(0.50, y_theta[1], color='red', alpha=0.1)          # strong

    ax1.set_title(f"PDM — {choice}")
    ax1.set_xlabel("Period [days]")
    ax1.set_ylabel("Theta")
    ax1.set_ylim(0.55, 1.03)
    ax1.set_xlim(0, 10)

    # --- Phase ---
    ax2.scatter(phase_s, flux_s, s=5, color='black', alpha=0.4)
    ax2.set_title(f"Phase — {choice} (P={best_period:.3f} d)")
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Flux")

    style_axes(ax1)
    style_axes(ax2)

    plt.tight_layout()
    plt.show()
    
    # plt.savefig(fr"C:\Users\Ayush Shivkumar\OneDrive\Desktop\major_project\Datasets\TTau_{choice}_PDM_Phase.png", dpi = 500, 
    #             bbox_inches = 'tight')
# plt.show()