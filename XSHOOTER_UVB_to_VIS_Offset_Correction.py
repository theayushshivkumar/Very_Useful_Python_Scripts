# To align different arms of VLT/X-shooter in case of flux offsets.
# Here, it is done for aligning the UVB arm to the VIS arm, and can be tweaked for VIS-NIR arm
# offset correction as well.

def align_uvb_to_vis(wave, flux, uvb_range=(5200, 5560), vis_range=(5560, 5800),
                     overlap_mask=(5337,5560), poly_order=2, debug_plot=True):
    """
    Aligns UVB arm flux to VIS arm flux by fitting polynomials to clean regions
    and scaling UVB down by the vertical offset.

    Parameters
    ----------
    wave : array
        Wavelength array (Å).
    flux : array
        Flux array (same length as wave).
    uvb_range : tuple
        Wavelength range for UVB polynomial fit.
    vis_range : tuple
        Wavelength range for VIS polynomial fit.
    overlap_mask : tuple
        Wavelength range to ignore (noisy overlap).
    poly_order : int
        Order of polynomial fit (default=2).
    debug_plot : bool
        If True, plots fits before and after correction.

    Returns
    -------
    flux_corrected : array
        Flux array with UVB scaled to VIS.
    scale_factor : float
        Multiplicative scaling applied to UVB.
    """

    w = np.asarray(wave)
    f = np.asarray(flux)

    # masks
    uvb_mask = (w >= uvb_range[0]) & (w <= uvb_range[1])
    vis_mask = (w >= vis_range[0]) & (w <= vis_range[1])
    overlap_mask_bool = (w >= overlap_mask[0]) & (w <= overlap_mask[1])

    # fit polynomials
    p_uvb = np.polyfit(w[uvb_mask], f[uvb_mask], deg=poly_order)
    p_vis = np.polyfit(w[vis_mask], f[vis_mask], deg=poly_order)

    poly_uvb = np.poly1d(p_uvb)
    poly_vis = np.poly1d(p_vis)

    # pick reference wavelength = midpoint between UVB and VIS regions
    ref_wave = 0.5 * (uvb_range[1] + vis_range[0])

    f_uvb_ref = poly_uvb(ref_wave)
    f_vis_ref = poly_vis(ref_wave)

    # scale factor to bring UVB down to VIS
    scale_factor = f_vis_ref / f_uvb_ref

    flux_corrected = f.copy()
    flux_corrected[w <= uvb_range[1]] *= scale_factor  # scale only UVB part

    if debug_plot:
        plt.figure(figsize=(12,6))
        plt.plot(w, f, label="Original spectrum", lw=0.7, alpha=0.7)
        plt.plot(w, flux_corrected, label="Corrected spectrum", lw=0.7)

        # show fits
        plt.plot(w, poly_uvb(w), 'r--', label="UVB poly fit")
        plt.plot(w, poly_vis(w), 'g--', label="VIS poly fit")

        # mark reference point
        plt.axvline(ref_wave, color='k', ls="--", alpha=0.5)
        plt.scatter(ref_wave, f_uvb_ref, color='red')
        plt.scatter(ref_wave, f_vis_ref, color='green')

        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Flux")
        plt.legend()
        plt.title("UVB–VIS Arm Alignment")
        plt.grid(alpha=0.3)
        plt.show()

    return flux_corrected, scale_factor

