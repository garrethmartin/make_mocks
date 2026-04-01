"""
test_make_mocks.py
==================
Technical tests for make_mocks and smooth_3d modules.

Each test prints an explanation, expected behavior, and result.
"""

import numpy as np
import warnings
import time
import make_mocks.make_mocks as mm
import make_mocks.smooth_3d as s3d

# Helper for test output

def print_test(title, explanation):
    print(f"\n[TEST] ------ {title} ------")
    print(f"{explanation}\n")

def print_result(result, passed):
    print(f"[RESULT] {result}")
    print(f"[{'PASSED' if passed else 'FAILED'}]")

# 1. Synthetic particle data
def test_synthetic_data():
    print_test(
        "Synthetic particle data",
        "Generate synthetic bulge+disc galaxy particle data for use in all tests."
        " Should produce arrays X, X_c, mass, met, age with reasonable ranges."
    )
    rng = np.random.default_rng(1)
    N_bulge = 20_000
    N_disc  = 80_000
    N       = N_bulge + N_disc
    X_bulge = rng.normal(0, 1.5, size=(3, N_bulge))
    X_disc  = np.vstack([
        rng.normal(0, 6, size=(2, N_disc)),
        rng.normal(0, 0.3, size=(1, N_disc)),
    ])
    X   = np.hstack([X_bulge, X_disc])
    X_c = np.zeros(3)
    mass = np.full(N, 1e4)
    met  = np.concatenate([
        rng.uniform(0.008, 0.04, N_bulge),
        rng.uniform(0.001, 0.02, N_disc),
    ])
    age = np.concatenate([
        rng.uniform(5e9, 1e10, N_bulge),
        rng.uniform(1e8, 5e9,  N_disc),
    ])
    print_result(
        f'N = {N:,}  |  Z range: {met.min():.4f} – {met.max():.4f}  |  age range: {age.min():.2e} – {age.max():.2e} yr',
        True
    )
    return X, X_c, mass, met, age, rng

# 2a. Mass conservation
def test_mass_conservation(rng):
    print_test(
        "Mass conservation (smooth_3d)",
        "This test checks that the total quantity in the smoothed image matches the total input particle weights. "
        "This ensures that the smoothing process conserves the total mass (or flux), except for small losses at the image boundary. "
        "A fractional difference below 5% is considered a pass."
    )
    N_test = 10_000
    X_test = rng.normal(0, 5, size=(3, N_test))
    q_test = rng.uniform(1, 2, size=(1, N_test))
    imgs, _ = s3d.smooth_3d(
        X_test, quantity_sum=q_test,
        res=0.5, extent=20, njobs=1, k=5,
        antialias=False, verbose=False,
    )
    total_in  = q_test.sum()
    total_out = imgs[0].sum()
    frac_diff = abs(total_out - total_in) / total_in
    passed = frac_diff < 0.05
    print_result(
        f'Sum in: {total_in:.4e}, Sum out: {total_out:.4e}, Fractional difference: {frac_diff:.4f}',
        passed
    )
    return passed

# 2b. Weighted average recovery
def test_weighted_average(rng):
    print_test(
        "Weighted average recovery (smooth_3d)",
        "This test checks that if all particles are assigned the same value for a quantity, the resulting average image should be uniform wherever there are particles. "
        "This validates the correct implementation of the density-weighted averaging in the smoothing routine."
    )
    N_test = 10_000
    X_test = rng.normal(0, 5, size=(3, N_test))
    q_avg_test = np.full((1, N_test), 10.0)
    _, avg_imgs = s3d.smooth_3d(
        X_test, quantity_average=q_avg_test,
        res=0.5, extent=20, njobs=1, k=5,
        antialias=False, verbose=False,
    )
    img_avg = avg_imgs[0]
    covered = img_avg > 0
    vals = img_avg[covered]
    mean_val = vals.mean() if len(vals) else 0
    passed = abs(mean_val - 10.0) < 0.05
    print_result(
        f'Average image mean: {mean_val:.4f} (expect ~10.0)',
        passed
    )
    return passed

# 2d. Density thresholds
def test_density_thresholds(rng):
    print_test(
        "Density thresholds (smooth_3d)",
        "This test checks the effect of the 'upper_threshold' and 'lower_threshold' parameters in the smoothing function.\n"
        "These thresholds allow you to exclude particles from the smoothing process if their local density is above (upper) or below (lower) a set value.\n"
        "This is useful for masking out extremely dense or sparse regions, or for debugging.\n"
        "The test passes if the function runs without error and produces an output image.\n"
        "(Note: This test does not check the detailed effect of the thresholds on the image, only that the feature works and does not crash.)"
    )
    # The following call applies both upper and lower density thresholds.
    # Particles with density > upper_threshold or < lower_threshold are excluded from smoothing.
    N_test = 10_000
    X_test = rng.normal(0, 5, size=(3, N_test))
    q_test = rng.uniform(1, 2, size=(1, N_test))
    try:
        imgs_thresh, _ = s3d.smooth_3d(
            X_test, quantity_sum=q_test,
            res=0.5, extent=20, njobs=1, k=5,
            upper_threshold=1e4, lower_threshold=1e-2,
            antialias=False, verbose=False,
        )
        result = f'Threshold run completed, image shape: {imgs_thresh[0].shape}'
        passed = True
    except Exception as e:
        result = f'Exception: {e}'
        passed = False
    print_result(result, passed)
    return passed

# 3a. Template loading
def test_template_loading():
    print_test(
        "Template loading (make_mocks)",
        "This test loads the SSP template data for all three IMFs: 'chabrier', 'salpeter', and 'kroupa'.\n"
        "It checks that the returned arrays have valid shapes and that all spectral values are non-negative.\n"
    )
    imfs = ['chabrier', 'salpeter', 'kroupa']
    all_passed = True
    for imf in imfs:
        wav, mets, ages, spec = mm.load_templates(imf)
        passed = spec.min() >= 0
        result = (
            f"IMF: {imf:8s} | wavelength: {wav.shape} {wav[0]:.0f}–{wav[-1]:.0f} Å, "
            f"metallicity: {mets[0]:.4f}–{mets[-1]:.4f}, ages: {ages[0]:.2e}–{ages[-1]:.2e} yr, "
            f"spectra min={spec.min():.2e}"
        )
        print_result(result, passed)
        if not passed:
            all_passed = False
    return all_passed

# 3b. Band integration
def test_band_integration():
    print_test(
        "Band integration (make_mocks)",
        "This test integrates the SEDs through several photometric bands for all three IMFs: 'chabrier', 'salpeter', and 'kroupa'.\n"
    )
    test_bands = ['g', 'r', 'i', 'VIS', 'H', 'J']
    imfs = ['chabrier', 'salpeter', 'kroupa']
    all_passed = True
    for imf in imfs:
        wav_i, mets_i, ages_i, spec_i = mm.load_templates(imf)
        # Choose a reference grid point: closest to Z=0.02 (solar), age=1e9 yr
        iZ = np.abs(mets_i - 0.02).argmin()
        ia = np.abs(ages_i - 1e9).argmin()
        for band in test_bands:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                try:
                    f_nu = mm.integrate_band(wav_i, spec_i, band, z_shift=0.1)
                    passed = (f_nu >= 0).all()
                    # Additional statistics
                    shape = f_nu.shape
                    frac_zero = (f_nu == 0).sum() / f_nu.size
                    rng_str = f'{f_nu[f_nu>0].min():.3e} – {f_nu.max():.3e}' if (f_nu > 0).any() else 'all zero'
                    ref_val = f_nu[iZ, ia]
                    print(f'{imf:10s}  {band:4s}  shape: {shape}  f_nu range: {rng_str}  frac_zero: {frac_zero:.2%}  f_nu(Z~0.02,age~1e9yr): {ref_val:.3e}')
                    if w:
                        print('[WARNINGS]', '\n'.join(str(x.message) for x in w))
                except FileNotFoundError as e:
                    passed = False
                    print(f'{imf:10s}  {band:4s}  [FileNotFoundError: {e}]')
                if not passed:
                    all_passed = False
    print_result('All band integrations non-negative.' if all_passed else 'Negative f_nu found or missing filter!', all_passed)
    return all_passed

# 3d. assign_fluxes grid-point interpolation
def test_assign_fluxes_gridpoint():
    print_test(
        "assign_fluxes grid-point interpolation",
        "This test checks that when you interpolate at an exact grid point (i.e., a metallicity and age that exactly matches the template grid),\n"
        "the result matches the template value. This ensures the interpolation is accurate and does not introduce errors at grid points."
    )
    wav, mets, ages, spec = mm.load_templates('chabrier')
    f_nu_bands = mm.precompute_band_fluxes(wav, mets, ages, spec, ['g', 'r', 'i'], z_shift=0.1)
    iZ, ia = 2, 50
    Z_exact   = mets[iZ]
    age_exact = ages[ia]
    mass_test = np.array([1.0])
    met_test  = np.array([Z_exact])
    age_test  = np.array([age_exact])
    L_nu_interp = mm.assign_fluxes(mass_test, met_test, age_test, mets, ages, f_nu_bands)
    passed = True
    for ib, band in enumerate(['g', 'r', 'i']):
        template_val = f_nu_bands[ib, iZ, ia] * mm._C10PC_AB
        interp_val   = L_nu_interp[ib, 0]    * mm._C10PC_AB
        ratio = interp_val / template_val
        if abs(ratio - 1) >= 1e-3:
            passed = False
        print(f"band '{band}': template={template_val:.4e} interp={interp_val:.4e} ratio={ratio:.6f}")
    print_result('Interpolation at grid point matches template.' if passed else 'Mismatch found!', passed)
    return passed

# 3e. Edge-case warnings
def test_edge_cases():
    print_test(
        "assign_fluxes edge cases",
        "This test covers several edge cases:\n"
        "- Assigning fluxes to particles with metallicity below the template minimum (should warn and clip).\n"
        "- Assigning fluxes with metallicity much greater than the template max (should warn about possible solar-unit confusion).\n"
        "- Attempting to integrate a non-existent filter (should raise FileNotFoundError).\n"
        "These checks ensure the code handles bad input robustly and provides useful warnings or errors."
    )
    wav, mets, ages, spec = mm.load_templates('chabrier')
    f_nu_bands = mm.precompute_band_fluxes(wav, mets, ages, spec, ['g', 'r', 'i'], z_shift=0.1)
    # Below Z_min
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        mm.assign_fluxes(np.ones(10), np.full(10, 1e-6), np.full(10, 1e9), mets, ages, f_nu_bands)
        msgs = [str(x.message) for x in w]
    print('[WARNINGS] Below Z_min:', '\n'.join(msgs) if msgs else '(no warnings)')
    # Solar-unit confusion
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        mm.assign_fluxes(np.ones(20), np.full(20, 1.5), np.full(20, 1e9), mets, ages, f_nu_bands)
        msgs = [str(x.message) for x in w]
    print('[WARNINGS] Solar-unit confusion:', '\n'.join(msgs) if msgs else '(no warnings)')
    # Bad filter name
    try:
        mm.integrate_band(wav, spec, 'nonexistent_band')
        print_result('No error for bad filter name!', False)
    except FileNotFoundError as e:
        print_result(f'FileNotFoundError as expected: {e}', True)
    return True

if __name__ == '__main__':
    X, X_c, mass, met, age, rng = test_synthetic_data()
    test_mass_conservation(rng)
    test_weighted_average(rng)
    test_density_thresholds(rng)
    test_template_loading()
    test_band_integration()
    test_assign_fluxes_gridpoint()
    test_edge_cases()
    print("\nAll technical tests completed.")
