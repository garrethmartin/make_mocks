
"""
make_mocks.py
=============
Mock galaxy imaging pipeline using SSP templates from BC03 (Bruzual & Charlot 2003, updated 2013) and E-MILES (Vazdekis et al. 2016, pPXF v9.0).

Unit conventions
----------------
Positions      : proper kpc
Masses         : M_sun  (initial stellar mass at formation)
Metallicity    : absolute mass fraction Z  (solar about 0.02)
Ages           : yr
Wavelengths    : Angstrom
Spectra        : L_sun / Ang / M_sun   (from HDF5 templates)
Band luminosity: L_sun / Hz / M_sun  (after filter integration)
Image pixels   : AB flux units -- pixel_value = F_nu x 10^(48.6/2.5)
                 so that  m_AB = -2.5 x log10(pixel_value)

Pipeline
--------
1. load_templates()            -- load SSP HDF5 file (BC03 or E-MILES)
2. precompute_band_fluxes()    -- integrate SEDs through photometric filters
3. assign_fluxes()             -- interpolate grid to particles; scale by mass
4. make_image()                -- smooth, distance modulus, PSF, noise (optional)

Quick-start
-----------
>>> import make_mocks as mm
>>> # Noiseless (mu_lims=None is the default):
>>> imgs_noisy, imgs_clean = mm.make_image(
...     X, X_c, mass, metallicity, age,
...     psfs=[psf_g, psf_r, psf_i],
...     bands=['g', 'r', 'i'],
...     z_fix=0.1,
... )
>>> # imgs_noisy is None; imgs_clean holds the noiseless images.
>>>
>>> # With noise (pass a limiting magnitude per band):
>>> imgs_noisy, imgs_clean = mm.make_image(
...     X, X_c, mass, metallicity, age,
...     mu_lims=[29.5, 30.0, 30.5],
...     psfs=[psf_g, psf_r, psf_i],
...     bands=['g', 'r', 'i'],
...     z_fix=0.1,
... )

Notes
-----
- Redshifting: the SED is shifted to the observer frame before filter integration:
    F_obs(lam_obs) = F_rest(lam_rest) / (1+z), with  lam_obs = lam_rest * (1+z).
- Reference distance: fluxes are first computed at the 10-pc reference distance, then rescaled to the
    luminosity distance via the distance modulus.
- Metallicity convention: this module uses absolute mass fractions throughout (e.g. solar Z_sun ~ 0.02).
    If the simulation data stores metallicity in solar units (Z / Z_sun), multiply by 0.02 before calling
    any function here.
- IMF / template options:
    'chabrier' -- BC03, Padova 1994 isochrones, BaSeL 3.1 (default)
    'salpeter' -- BC03, Padova 1994 isochrones, BaSeL 3.1
    'kroupa'   -- E-MILES v9.0, BaSTI isochrones, Kroupa revised IMF
- Template grid boundaries cover different metallicity and age ranges depending on the IMF / template set.:
    Particles outside these ranges are clipped; a [WARNING] is issued.
"""

import os
import warnings
import numpy as np
import h5py
from . import smooth_3d


# Physical constants

L_SUN = 3.828e33    # solar luminosity [erg/s]
PC_CM = 3.0857e18   # 1 parsec [cm]
C_CGS = 2.998e10    # speed of light [cm/s]

# F_nu [erg/s/cm^2/Hz] = L_nu [L_sun/Hz]  *  _FLUX_10PC
_FLUX_10PC = L_SUN / (4.0 * np.pi * (10.0 * PC_CM) ** 2)

# AB zero-point: multiply F_nu [erg/s/cm^2/Hz] by this to get AB flux units
# where m_AB = -2.5 * log10(AB_flux)
_AB_ZEROPOINT = 10.0 ** (48.6 / 2.5)

# Combined: AB flux units at 10 pc = L_nu [L_sun/Hz]  *  _C10PC_AB
_C10PC_AB = _FLUX_10PC * _AB_ZEROPOINT

_MSUN_G = 1.989e33           # solar mass [g]
_KPC_CM = 1e3 * PC_CM        # 1 kpc [cm]

# numpy >= 2.0 replaced np.trapz with np.trapezoid
# This should keep the code happy no matter the numpy version
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

# Default path resolution

_HERE = os.path.dirname(os.path.abspath(__file__))

_DUST_OPACITY_FILE = os.path.join(
    _HERE, 'templates', 'filters', 'kext_albedo_WD_MW_3.1_60.txt'
)

_IMF_FILES = {
    'chabrier': 'bc03_chabrier.hdf5',        # BC03, Padova 1994, BaSeL 3.1
    'salpeter': 'bc03_salpeter.hdf5',        # BC03, Padova 1994, BaSeL 3.1
    'kroupa':   'emiles_kroupa.hdf5',        # E-MILES v9.0, BaSTI isochrones, Kroupa revised IMF
}

_VALID_IMFS = tuple(_IMF_FILES)


def _template_path(imf, template_dir=None):
    d = template_dir or os.path.join(_HERE, 'templates')
    fname = _IMF_FILES.get(imf.lower())
    if fname is None:
        raise ValueError(
            f"imf must be one of {_VALID_IMFS}, got {imf!r}"
        )
    return os.path.join(d, fname)


def _filter_path(band, filter_dir=None):
    d = filter_dir or os.path.join(_HERE, 'templates', 'filters')
    return os.path.join(d, band)


# Template loading

def load_templates(imf='chabrier', template_dir=None):
    """
    Load SSP templates from HDF5.

    Parameters:
        imf: 'chabrier', 'salpeter', or 'kroupa'.
            Stellar IMF / template set.
        template_dir: str, optional
            Directory containing the HDF5 files. Defaults to ./templates/ relative to this module.

    Returns:
        wavelength: ndarray, shape (n_wav,)
            Wavelength grid [Angstrom].
        metallicities: ndarray, shape (n_Z,)
            Metallicity grid [absolute mass fraction; solar about 0.02].
        ages: ndarray, shape (n_ages,)
            Age grid [yr]. BC03 templates have an age=0 first entry which is dropped by assign_fluxes.
             E-MILES has no age=0 entry.
        spectra: ndarray, shape (n_Z, n_ages, n_wav)
            SSP specific luminosity [L_sun / Ang / M_sun].
    """
    import h5py
    with h5py.File(_template_path(imf, template_dir), 'r') as f:
        wavelength    = f['wavelength'][()]
        metallicities = f['metallicities'][()]
        ages          = f['ages'][()]
        spectra       = f['spectra'][()]
    return wavelength, metallicities, ages, spectra


# Filter integration

def integrate_band(wavelength, spectra, band, z_shift=0.0, filter_dir=None):
    """Integrate SSP SEDs through a photometric filter.

    The SED is redshifted to the observer frame before integration, using
    the photon-counting (AB) convention::

        f_nu = integral(T(lam) F_lam(lam) / nu  dlam)
               / integral(T(lam) / lam  dlam)

    where lam is the observer-frame wavelength and nu = c / lam.

    Parameters
    ----------
    wavelength : ndarray, shape (n_wav,)
        Rest-frame wavelength grid [Ang].
    spectra : ndarray, shape (n_Z, n_ages, n_wav)
        SSP specific luminosity [L_sun / Ang / M_sun].
    band : str
        Filter name (filename in filter_dir, e.g. 'g', 'VIS', 'H').
    z_shift : float
        Source redshift.
    filter_dir : str, optional
        Directory containing filter transmission files.

    Returns
    -------
    f_nu : ndarray, shape (n_Z, n_ages)
        Band-averaged specific luminosity [L_sun / Hz / M_sun].
        Returns zeros if the SED has no overlap with the filter.

    Warns
    -----
    If the SED covers less than 80% of the filter bandpass, the integrated
    flux may be underestimated.  If there is no overlap at all, zeros are
    returned.
    """
    filter_file = _filter_path(band, filter_dir)
    try:
        fdata = np.loadtxt(filter_file)
    except OSError:
        raise FileNotFoundError(
            f"Filter file not found: {filter_file!r}.  "
            f"Available filters are in {os.path.dirname(filter_file)!r}."
        )
    lambda_f, T_f = fdata[:, 0], fdata[:, 1]

    # Observer-frame wavelengths and energy-dimming factor
    lambda_obs  = wavelength * (1.0 + z_shift)
    energy_damp = 1.0 / (1.0 + z_shift)

    # Restrict to the wavelengths covered by the filter
    mask = (lambda_obs >= lambda_f[0]) & (lambda_obs <= lambda_f[-1])
    lam  = lambda_obs[mask]

    if lam.size < 2:
        warnings.warn(
            f"[WARNING] Band '{band}' (z={z_shift:.3f}): No overlap between the redshifted SED "
            f"({lambda_obs[0]:.0f}--{lambda_obs[-1]:.0f} Ang) and the filter "
            f"({lambda_f[0]:.0f}--{lambda_f[-1]:.0f} Ang). Returning zero flux. Check wavelength ranges!",
            stacklevel=2,
        )
        return np.zeros(spectra.shape[:2])

    # Warn if coverage of the filter is significantly incomplete.
    # This matters most for long-wavelength filters at high redshift
    filter_width  = lambda_f[-1] - lambda_f[0]
    coverage_frac = (lam[-1] - lam[0]) / filter_width
    if coverage_frac < 0.8:
        warnings.warn(
            f"[WARNING] Band '{band}' (z={z_shift:.3f}): SED covers only "
            f"{100 * coverage_frac:.0f}% of the filter bandpass "
            f"(SED in-band: {lam[0]:.0f}--{lam[-1]:.0f} Ang; "
            f"filter: {lambda_f[0]:.0f}--{lambda_f[-1]:.0f} Ang). "
            f"Band flux may be inaccurate. Don't blame me if your results look weird!",
            stacklevel=2,
        )

    # Interpolate filter transmission onto the SED wavelength grid
    T  = np.interp(lam, lambda_f, T_f)       # (n_lam,)
    nu = C_CGS / (lam * 1e-8)                # Hz; 1e-8 converts Ang -> cm

    # Observer-frame SED: F_lam(lam_obs) = F_lam(lam_rest) / (1+z)
    F = spectra[:, :, mask] * energy_damp    # (n_Z, n_ages, n_lam)

    # Photon-counting normalisation: integral(T / lam  dlam)
    norm = _trapz(T / lam, lam)
    if norm < 1e-30:
        warnings.warn(
            f"[WARNING] Band '{band}': Filter normalisation is basically zero (norm = {norm:.2e}). "
            f"The filter may have no valid transmission over the SED wavelength range. Returning zero flux. Maybe check your filter file?",
            stacklevel=2,
        )
        return np.zeros(spectra.shape[:2])

    # f_nu = integral(T * F_lam / nu / norm  dlam)  [L_sun / Hz / M_sun]
    # T and nu broadcast over the first two (Z, age) axes of F.
    f_nu = _trapz(F * (T / (nu * norm)), lam, axis=-1)   # (n_Z, n_ages)

    # Physical SEDs must give non-negative f_nu.  Flag and clip any that don't.
    n_neg = int((f_nu < 0).sum())
    if n_neg:
        warnings.warn(
            f"[WARNING] Band '{band}': {n_neg} (Z, age) grid point(s) produced negative f_nu after integration (min = {f_nu.min():.2e}). "
            f"This is just a numerical hiccup; values are clipped to zero. It's probably fine.",
            stacklevel=2,
        )
        f_nu = np.maximum(f_nu, 0.0)

    # Warn if any grid point has exactly zero flux
    n_zero = int((f_nu == 0).sum())
    total = f_nu.size
    if 0 < n_zero < total:
        warnings.warn(
            f"[WARNING] Band '{band}': {n_zero} of {total} (Z, age) grid points have exactly zero flux. "
            f"This may mean the SED does not cover the filter for some templates.",
            stacklevel=2,
        )
    return f_nu


def precompute_band_fluxes(wavelength, metallicities, ages, spectra,
                           bands, z_shift=0.0, filter_dir=None, verbose=False):
    """Integrate SSP SEDs through multiple filters on the full (Z, age) grid.

    Parameters
    ----------
    wavelength, metallicities, ages, spectra
        As returned by :func:`load_templates`.
    bands : list of str
        Filter names.
    z_shift : float
        Source redshift.
    filter_dir : str, optional
        Directory containing filter transmission files.
    verbose : bool

    Returns
    -------
    f_nu_bands : ndarray, shape (n_bands, n_Z, n_ages)
        Band-averaged specific luminosity [L_sun / Hz / M_sun].
    """
    f_nu_list = []
    for band in bands:
        if verbose:
            print(f'[INFO] Integrating SED through band {band!r}...')
        f_nu_list.append(
            integrate_band(wavelength, spectra, band,
                           z_shift=z_shift, filter_dir=filter_dir)
        )
    return np.array(f_nu_list)   # (n_bands, n_Z, n_ages)


#  Particle flux assignment

def assign_fluxes(mass, metallicity, age, metallicities, ages, f_nu_bands):
    """Assign per-particle band luminosities by interpolating the SSP grid.

    Interpolation is bilinear in log(Z)--log(age) space. The BC03 age=0 yr entry
    is excluded; the minimum interpolation age is 1e5 yr.

    Out-of-range particles are clipped to the nearest grid boundary and a
    warning is issued.

    Parameters
    ----------
    mass : ndarray, shape (n_p,)
        Initial stellar mass [M_sun].
    metallicity : ndarray, shape (n_p,)
        Metallicity [absolute mass fraction; solar approx 0.02].
    age : ndarray, shape (n_p,)
        Stellar age [yr].
    metallicities : ndarray, shape (n_Z,)
        Template metallicity grid [absolute mass fraction].
    ages : ndarray, shape (n_ages,)
        Template age grid [yr].  The age = 0 entry is excluded automatically.
    f_nu_bands : ndarray, shape (n_bands, n_Z, n_ages)
        Template band fluxes [L_sun / Hz / M_sun], from
        :func:`precompute_band_fluxes`.

    Returns
    -------
    L_nu : ndarray, shape (n_bands, n_p)
        Per-particle specific luminosity [L_sun / Hz].

    Warns
    -----
    * Non-finite or non-positive metallicity/age values.
    * Particles clipped at the Z or age grid boundaries.
    * If more than half the particles have Z > Z_max, possible unit confusion
      hint issued.
    """
    from scipy.interpolate import RegularGridInterpolator

    mass        = np.asarray(mass,        dtype=float).ravel()
    metallicity = np.asarray(metallicity, dtype=float).ravel()
    age         = np.asarray(age,         dtype=float).ravel()
    n_p         = len(mass)

    # Validate inputs

    bad_Z = ~np.isfinite(metallicity) | (metallicity <= 0)
    if bad_Z.any():
        warnings.warn(
            f"[WARNING] assign_fluxes: {bad_Z.sum()}/{n_p} particles have non-finite or non-positive metallicity. "
            f"Replacing with Z_min = {metallicities[0]:.4g}.",
            stacklevel=2,
        )
    metallicity = np.where(bad_Z, metallicities[0], metallicity)

    bad_age = ~np.isfinite(age) | (age < 0)
    if bad_age.any():
        warnings.warn(
            f"[WARNING] assign_fluxes: {bad_age.sum()}/{n_p} particles have non-finite or negative age. "
            f"Replacing with age = 0 (will be clipped to age_min). Time travel not supported.",
            stacklevel=2,
        )
    age = np.where(bad_age, 0.0, age)

    # Build interpolation grid (drop the age=0 yr entry)

    # BC03 age[0] = 0 yr has identical spectra to age[1] = 1e5 yr, so dropping
    # it avoids log10(0) while losing no information.
    age_mask  = ages > 0
    age_grid  = ages[age_mask]                        # (n_ages_pos,)
    f_nu_grid = f_nu_bands[:, :, age_mask]            # (n_bands, n_Z, n_ages_pos)

    Z_min,   Z_max   = metallicities[0], metallicities[-1]
    age_min, age_max = age_grid[0],      age_grid[-1]

    # Report boundary clipping

    n_Z_lo = int((metallicity < Z_min).sum())
    n_Z_hi = int((metallicity > Z_max).sum())
    n_a_lo = int((age < age_min).sum())
    n_a_hi = int((age > age_max).sum())

    if n_Z_lo:
        bad = metallicity[metallicity < Z_min]
        warnings.warn(
            f"[WARNING] assign_fluxes: {n_Z_lo}/{n_p} ({100*n_Z_lo/n_p:.1f}%) particles have Z < Z_min ({Z_min:.4g}). "
            f"Clipped to Z_min. [Particle Z range: {bad.min():.3e}--{bad.max():.3e}]",
            stacklevel=2,
        )
    if n_Z_hi:
        bad = metallicity[metallicity > Z_max]
        warnings.warn(
            f"[WARNING] assign_fluxes: {n_Z_hi}/{n_p} ({100*n_Z_hi/n_p:.1f}%) particles have Z > Z_max ({Z_max:.4g}). "
            f"Clipped to Z_max. [Particle Z range: {bad.min():.3e}--{bad.max():.3e}]",
            stacklevel=2,
        )
        # Heuristic: if most particles exceed Z_max, we might have passed solar-normalised metallicities instead of absolute fractions.
        if n_Z_hi > 0.5 * n_p:
            warnings.warn(
                f"[WARNING] assign_fluxes: More than half of all particles have Z > Z_max ({Z_max:.4g}). "
                f"Did you pass metallicity in solar units (Z / Z_sun)? If so, multiply by 0.02 to get absolute fractions.",
                stacklevel=2,
            )
    if n_a_lo:
        bad = age[age < age_min]
        warnings.warn(
            f"[WARNING] assign_fluxes: {n_a_lo}/{n_p} ({100*n_a_lo/n_p:.1f}%) particles have age < age_min ({age_min:.3e} yr). "
            f"Clipped to age_min. [Particle age range: {bad.min():.3e}--{bad.max():.3e} yr]",
            stacklevel=2,
        )
    if n_a_hi:
        bad = age[age > age_max]
        warnings.warn(
            f"[WARNING] assign_fluxes: {n_a_hi}/{n_p} ({100*n_a_hi/n_p:.1f}%) particles have age > age_max ({age_max:.3e} yr). "
            f"Clipped to age_max. [Particle age range: {bad.min():.3e}--{bad.max():.3e} yr]",
            stacklevel=2,
        )

    Z_c   = np.clip(metallicity, Z_min,   Z_max)
    age_c = np.clip(age,         age_min, age_max)

    # Interpolate in log(Z)--log(age) space

    log_Z_grid   = np.log10(metallicities)
    log_age_grid = np.log10(age_grid)
    pts          = np.column_stack([np.log10(Z_c), np.log10(age_c)])  # (n_p, 2)

    L_nu = []
    for f_nu in f_nu_grid:    # iterate over bands: f_nu is (n_Z, n_ages_pos)
        # Interpolate log10(flux) to avoid negative artefacts.
        # A floor of 1e-40 to handle any genuine zeros in the template
        # (e.g. outside the filter wavelength range).
        log_f  = np.log10(np.maximum(f_nu, 1e-40))
        interp = RegularGridInterpolator(
            (log_Z_grid, log_age_grid), log_f,
            method='linear',
            bounds_error=False,   # should never happen after clipping above
            fill_value=None,
        )
        L_nu.append(10.0 ** interp(pts) * mass)   # (n_p,)  [L_sun / Hz]

    return np.array(L_nu)   # (n_bands, n_p)


# All-in-one flux calculation

def calculate_fluxes(X, X_c, mass, metallicity, age, z_fix, bands,
                     imf='chabrier', template_dir=None, filter_dir=None,
                     verbose=False):
    """Calculate per-particle flux densities in AB units at the 10-pc reference.

    Loads the templates, integrates through the requested filters, and
    interpolates to each particle's metallicity and age.

    Parameters
    ----------
    X : ndarray, shape (3, n_p)
        Particle positions [proper kpc].
    X_c : ndarray, shape (3,)
        Galaxy centre [proper kpc].  Positions are re-centred on X_c.
    mass : ndarray, shape (n_p,)
        Initial stellar mass [M_sun].
    metallicity : ndarray, shape (n_p,)
        Metallicity [absolute mass fraction; solar approx 0.02].
    age : ndarray, shape (n_p,)
        Stellar age [yr].
    z_fix : float
        Source redshift (SEDs are redshifted before filter integration).
    bands : list of str
        Filter names.
    imf : {'chabrier', 'salpeter'}
    template_dir, filter_dir : str, optional
        Override the default template / filter directories.
    verbose : bool

    Returns
    -------
    X_centred : ndarray, shape (3, n_p)
        Particle positions relative to X_c [proper kpc].
    fluxes : ndarray, shape (n_bands, n_p)
        AB flux units at the 10-pc reference distance.
        To get observed AB magnitude:
        m_AB = -2.5 * log10(flux) + distance_modulus(z_fix)
    """
    X_c   = np.asarray(X_c)
    X_cen = X - X_c[:, None]

    wav, mets, ages, spec = load_templates(imf, template_dir)

    f_nu_bands = precompute_band_fluxes(
        wav, mets, ages, spec, bands,
        z_shift=z_fix, filter_dir=filter_dir, verbose=verbose,
    )

    # L_nu [L_sun/Hz] -> AB flux units at 10 pc
    fluxes = assign_fluxes(mass, metallicity, age, mets, ages, f_nu_bands) * _C10PC_AB

    return X_cen, fluxes


# Cosmology helper

def _resolve_cosmology(cosmology):
    """Return an astropy cosmology object from a name string or object.

    Parameters
    ----------
    cosmology : str or astropy.cosmology.Cosmology
        Named cosmology (e.g. ``'WMAP7'``, ``'Planck18'``) or a pre-built
        astropy cosmology instance.  Available named cosmologies:
        WMAP5, WMAP7, WMAP9, Planck13, Planck15, Planck18.

    Returns
    -------
    astropy.cosmology.Cosmology
    """
    if isinstance(cosmology, str):
        from astropy.cosmology import available, default_cosmology
        if cosmology not in available:
            raise ValueError(
                f"Unknown cosmology name {cosmology!r}.  "
                f"Available names: {sorted(available)}"
            )
        from astropy import cosmology as _cosmo_module
        return getattr(_cosmo_module, cosmology)
    # Assume it is already an astropy cosmology object
    return cosmology


# Dust attenuation

def build_dust_cube(X_gas, mass_gas, metallicity_gas, extent, res,
                    dust_to_metal=0.4, k=5, n_resample=100, scatter_batch=1000, njobs=1):
    """
    Build a 3D dust mass density cube from gas particles.

    Computes dust mass per particle as m_dust = m_gas * Z * dust_to_metal, then delegates to smooth_3d.build_density_cube for the adaptive kNN Gaussian smoothing in 3D.

    Parameters:
        X_gas: array (3, n_gas)
            Gas particle positions [kpc], centered on the galaxy.
        mass_gas: array (n_gas,)
            Gas particle masses [M_sun].
        metallicity_gas: array (n_gas,)
            Gas metallicity [absolute mass fraction Z].
        extent: float
            Half-size of the cube [kpc]. The cube spans [-extent, +extent] in each axis.
        res: float
            Voxel size [kpc]. Does not need to match the image pixel scale.
        dust_to_metal: float
            Fraction of metals locked in dust grains (default 0.4).
        k: int
            k-NN neighbours for smoothing-length estimation.
        n_resample: int
            Sub-particles per particle for scatter smoothing (default 100).
        scatter_batch: int
            Mini-batch size for scatter smoothing (default 1000).
        njobs: int
            Parallel workers for the kNN step. More njobs means your computer will be more efficient at heating your room, and might also run faster!

    Returns:
        dust_cube: ndarray (n_pix, n_pix, n_pix), float32
            Dust mass density [M_sun / kpc^3].
    """
    m_dust = (np.asarray(mass_gas, dtype=np.float64).ravel()
              * np.asarray(metallicity_gas, dtype=np.float64).ravel()
              * dust_to_metal)
    return smooth_3d.build_density_cube(X_gas, m_dust, res=res, extent=extent,
                                   k=k, n_resample=n_resample,
                                   scatter_batch=scatter_batch, njobs=njobs)


def apply_dust_attenuation(X_star, fluxes, dust_cube, bands, z_fix, extent,
                            projection='xy', filter_dir=None,
                            dust_opacity_file=None):
    """Apply a dust-screen attenuation to per-particle fluxes.

    Integrates the dust density cube along the line of sight to produce a
    cumulative column density for each stellar particle, then attenuates each
    band by exp(-kappa(lambda_pivot) * Sigma_dust) where Sigma_dust is in g cm^-2.

    Parameters
    ----------
    X_star : array (3, n_p)
        Stellar particle positions [kpc], centred.
    fluxes : array (n_bands, n_p)
        Per-particle fluxes in AB units (from :func:`calculate_fluxes`).
    dust_cube : array (n_pix, n_pix, n_pix)
        3D dust mass density [M_sun / kpc^3] (from :func:`build_dust_cube`).
        Must share the same ``extent`` as ``X_star``.
    bands : list of str
        Filter names - must match the rows of fluxes.
    z_fix : float
        Source redshift; used to de-redshift filter pivot wavelengths.
    extent : float
        Half-size of the dust cube [kpc].
    projection : {'xy', 'xz', 'yz'}
        Projection axis; determines the line-of-sight direction.
    filter_dir : str, optional
        Override the default filter directory.
    dust_opacity_file : str, optional
        Path to a Weingartner & Draine (2001)-format opacity table.
        Default: ``templates/filters/kext_albedo_WD_MW_3.1_60.txt``.
        Columns (after 50 header rows): wavelength [um], ..., kappa_ext [cm^2/g].

    Returns
    -------
    fluxes_att : array (n_bands, n_p)
        Attenuated per-particle fluxes (same units and shape as ``fluxes``).
    """
    n_pix = dust_cube.shape[0]
    dz    = 2.0 * extent / n_pix   # voxel depth along LOS [kpc]

    if dust_opacity_file is None:
        dust_opacity_file = _DUST_OPACITY_FILE
    f_opac      = np.loadtxt(dust_opacity_file, skiprows=50)
    dust_lambda = f_opac[:, 0] * 1e4   # um to Angstrom
    dust_kappa  = f_opac[:, 4]          # cm^2 / g  (extinction opacity)

    # Rest-frame pivot wavelength for each band
    kappas = []
    for band in bands:
        fdata     = np.loadtxt(_filter_path(band, filter_dir))
        lam_pivot = np.average(fdata[:, 0], weights=fdata[:, 1]) / (1.0 + z_fix)
        kappas.append(float(np.interp(lam_pivot, dust_lambda, dust_kappa)))

    # Cumulative column density along projected axis: M_sun / kpc^2
    # col[i] = dust between the star at slice i and the observer (at +los_ax infinity),
    # i.e. total column minus the cumulative from the far edge up to slice i.
    los_ax        = {'xy': 2, 'xz': 1, 'yz': 0}[projection]
    col_total     = dust_cube.sum(axis=los_ax, keepdims=True) * dz
    col_msun_kpc2 = col_total - np.cumsum(dust_cube, axis=los_ax) * dz

    # Convert to g / cm^2
    col_gcm2 = col_msun_kpc2 * (_MSUN_G / _KPC_CM ** 2)

    # Map stellar particles to voxel indices
    edges = np.linspace(-extent, extent, n_pix + 1)
    def _idx(arr):
        return np.clip(np.digitize(arr, edges) - 1, 0, n_pix - 1)

    ix = _idx(X_star[0])
    iy = _idx(X_star[1])
    iz = _idx(X_star[2])

    sigma = col_gcm2[ix, iy, iz]   # (n_p,) - column density at each star's position

    fluxes_att = fluxes.copy()
    for i, kappa in enumerate(kappas):
        fluxes_att[i] *= np.exp(-kappa * sigma)

    return fluxes_att


# Image generation

def make_image(X, X_c, mass, metallicity, age, mu_lims=None, psfs=None,
               projection='xy', z_fix=0.1, imf='chabrier', extent=20,
               pixel_size=0.2, pixel_size_kpc=None,
               njobs=4, nsteps=500, n_resample=500, k=5,
               bands=None, template_dir=None, filter_dir=None,
               cosmology='WMAP7', dust_cube=None, dust_opacity_file=None,
               verbose=True):
    """Create mock galaxy images.

    Full pipeline:

    1. Compute per-particle flux in each band (AB units at 10-pc reference).
    2. Adaptively smooth particles into an image.
    3. Rescale flux from the 10-pc reference to the source luminosity distance.
    4. Convolve with the PSF.
    5. Add Gaussian noise calibrated to the limiting surface brightness.

    Parameters
    ----------
    X : ndarray, shape (3, n_p)
        Particle positions [proper kpc].
    X_c : ndarray, shape (3,)
        Galaxy centre [proper kpc].
    mass : ndarray, shape (n_p,)
        Initial stellar mass [M_sun].
    metallicity : ndarray, shape (n_p,)
        Metallicity [absolute mass fraction; solar approx 0.02].
    age : ndarray, shape (n_p,)
        Stellar age [yr].
    mu_lims : array-like, shape (n_bands,), optional
        3-sigma limiting AB magnitude in a 10" x 10" aperture, one per band.
        Pass ``None`` (default) to skip noise entirely; the first return value
        will be ``None``.
    psfs : list of 2-D ndarray, length n_bands
        PSF images, one per band (FFT-convolved with the image).
    projection : {'xy', 'xz', 'yz'}
        Line-of-sight projection axis.
    z_fix : float
        Source redshift.  When ``z_fix=0`` the distance-modulus rescaling is
        skipped and pixel values remain at the 10-pc reference flux, i.e. the
        absolute-magnitude flux scale (``m_AB`` gives the absolute AB magnitude).
        Use ``pixel_size_kpc`` in this case to avoid the undefined arcsec→kpc
        conversion.
    imf : {'chabrier', 'salpeter'}
        Stellar initial mass function.
    extent : float
        Image half-size [kpc].  Full image covers [-extent, +extent] per axis.
    pixel_size : float
        Angular pixel scale [arcsec].  Converted to kpc via the angular
        diameter distance at ``z_fix``.  Ignored when ``pixel_size_kpc`` is
        provided.  Also used for the noise formula regardless of
        ``pixel_size_kpc``.
    pixel_size_kpc : float, optional
        Physical pixel size [kpc].  When given, bypasses the arcsec→kpc
        conversion, allowing ``z_fix=0`` and direct physical-unit control.
        Takes precedence over ``pixel_size`` for the spatial grid only.
    njobs : int
        Number of parallel workers for adaptive smoothing.
    nsteps : int
        Density bins for adaptive smoothing (higher -> more accurate).
    n_resample : int
        Sub-particles per particle for kernel smoothing.
    k : int
        k-NN neighbours for density / smoothing-length estimate.
    bands : list of str, optional
        Filter names.  Defaults to ['g', 'r', 'i'].
    template_dir, filter_dir : str, optional
        Override the default template / filter directories.
    cosmology : str or astropy.cosmology.Cosmology, optional
        Cosmology used to convert pixel_size [arcsec] to physical [kpc] and
        to compute the distance modulus.  Pass a name string (e.g. ``'WMAP7'``,
        ``'Planck18'``) or a pre-built astropy cosmology object.
        Default: ``'WMAP7'``.
    dust_cube : ndarray (n_pix, n_pix, n_pix), optional
        Pre-built 3D dust mass density [M_sun / kpc^3] from
        :func:`build_dust_cube`.  When provided, a screen-model attenuation
        exp(-kappa * Sigma_dust) is applied to each particle's flux before
        spatial smoothing.  The cube must span [-extent, +extent] in all
        three axes.
    dust_opacity_file : str, optional
        Path to Weingartner & Draine (2001) opacity table.  Passed through
        to :func:`apply_dust_attenuation`; see that function for format details.
    verbose : bool

    Returns
    -------
    imgs : list of 2-D ndarray or None
        Noisy images (list of length n_bands), or ``None`` when
        ``mu_lims=None``.  Pixel values are AB flux units rescaled to the
        luminosity distance at ``z_fix``: ``m_AB = -2.5 * log10(pixel_value)``.
        At ``z_fix=0`` no rescaling is applied, so pixel values are at the
        10-pc reference distance and ``m_AB`` gives the absolute AB magnitude.
    imgs_clean : list of 2-D ndarray, length n_bands
        Noiseless images (same units).
    """
    if bands is None:
        bands = ['g', 'r', 'i']

    n_bands = len(bands)
    if psfs is None:
        raise ValueError("psfs must be provided (list of 2-D arrays, one per band).")
    if len(psfs) != n_bands:
        raise ValueError(
            f"len(psfs) = {len(psfs)} does not match len(bands) = {n_bands}."
        )
    if mu_lims is not None and len(mu_lims) != n_bands:
        raise ValueError(
            f"len(mu_lims) = {len(mu_lims)} does not match len(bands) = {n_bands}."
        )

    from scipy.signal import fftconvolve
    import importlib; s3d = importlib.import_module('make_mocks.smooth_3d')

    cosmo = _resolve_cosmology(cosmology)

    # Physical pixel scale: kpc per pixel
    if pixel_size_kpc is not None:
        res = float(pixel_size_kpc)
    else:
        if z_fix == 0.0:
            raise ValueError(
                "z_fix=0 makes arcsec -> kpc conversion undefined.  "
                "Provide pixel_size_kpc instead."
            )
        res = pixel_size / cosmo.arcsec_per_kpc_proper(z_fix).value

    # Per-particle fluxes in AB units at the 10-pc reference distance
    X_cen, fluxes = calculate_fluxes(
        X, X_c, mass, metallicity, age, z_fix, bands,
        imf=imf, template_dir=template_dir, filter_dir=filter_dir,
        verbose=verbose,
    )

    # Optional dust-screen attenuation
    if dust_cube is not None:
        fluxes = apply_dust_attenuation(
            X_cen, fluxes, dust_cube, bands, z_fix, extent,
            projection=projection, filter_dir=filter_dir,
            dust_opacity_file=dust_opacity_file,
        )

    # Adaptively smooth particles into images
    # If k=0 or None, do not smooth, just bin particles
    if k is None or k == 0:
        if verbose:
            print("[INFO] k=0 or None: no smoothing, using simple 2D histogram.")
        imgs = [s3d.bin_particles(X_cen, fluxes[i], projection=projection, res=res, extent=extent)
                for i in range(fluxes.shape[0])]
        _ = None
    else:
        imgs, _ = smooth_3d.smooth_3d(
            X_cen, fluxes,
            extent=extent, res=res, projection=projection,
            njobs=njobs, nsteps=nsteps, k=k, verbose=verbose,
        )

    # Rescale from 10-pc reference to the luminosity distance at z_fix.
    # distmod = 5 * log10(d_L / 10 pc), so 10^(-distmod/2.5) = (10 pc / d_L)^2.
    # At z=0, d_L=0 so distmod=-inf; skip the rescaling (fluxes stay at 10-pc).
    if z_fix > 0.0:
        dist_scale = 10.0 ** (-cosmo.distmod(z_fix).value / 2.5)
        imgs = [img * dist_scale for img in imgs]

    # PSF convolution
    imgs = [fftconvolve(img, psf, mode='same') for img, psf in zip(imgs, psfs)]

    imgs_clean = list(imgs)

    # Gaussian noise.
    # mu_lims[i] is the 3-sigma AB magnitude limit in a 10" x 10" aperture.
    # Corresponding flux limit (AB units) = 10^(-0.4 * mu).
    # Per-pixel noise sigma = flux_limit * pixel_size ["] * 10 ["] / 3
    if mu_lims is None:
        imgs_noisy = None
    else:
        imgs_noisy = [
            img + np.random.normal(
                loc=0.0,
                scale=(10.0 ** (-0.4 * mu) * pixel_size * 10.0) / 3.0,
                size=img.shape,
            )
            for img, mu in zip(imgs, mu_lims)
        ]

    if verbose:
        n_bands = len(bands)
        shape_str = imgs_clean[0].shape if imgs_clean and hasattr(imgs_clean[0], 'shape') else 'unknown'
        msg = f"[END] make_image: Created {n_bands} band(s) with image shape {shape_str}."
        if mu_lims is None:
            msg += " No noise added."
        else:
            msg += " Noise added with mu_lims=" + str(mu_lims)
        print(msg)
    return imgs_noisy, imgs_clean

# Spectral (IFU) functions

def _interp_weights(metallicity, age, metallicities, ages):
    """
    Bilinear interpolation corner indices and weights in log(Z)-log(age) space.

    Parameters
    ----------
    metallicity, age : arrays (n_p,)
    metallicities, ages : template grids (age = 0 entry already removed)

    Returns
    -------
    i_Z, i_Z1, i_a, i_a1 : int arrays (n_p,)   corner indices
    w_Z, w_a             : float32 arrays (n_p,) upper-corner weights in [0, 1]
    """
    log_Z_g   = np.log10(metallicities)
    log_age_g = np.log10(ages)
    log_Z_p   = np.log10(np.clip(metallicity, metallicities[0], metallicities[-1]))
    log_age_p = np.log10(np.clip(age,         ages[0],          ages[-1]))

    i_Z = np.clip(np.searchsorted(log_Z_g,   log_Z_p,   side='right') - 1,
                  0, len(log_Z_g)   - 2)
    i_a = np.clip(np.searchsorted(log_age_g, log_age_p, side='right') - 1,
                  0, len(log_age_g) - 2)
    i_Z1 = i_Z + 1
    i_a1 = i_a + 1

    dZ  = log_Z_g[i_Z1]   - log_Z_g[i_Z]
    da  = log_age_g[i_a1] - log_age_g[i_a]
    w_Z = np.where(dZ > 0, (log_Z_p   - log_Z_g[i_Z])   / dZ, 0.0)
    w_a = np.where(da > 0, (log_age_p - log_age_g[i_a]) / da, 0.0)

    return i_Z, i_Z1, i_a, i_a1, w_Z.astype(np.float32), w_a.astype(np.float32)


def assign_spectra(mass, metallicity, age, metallicities, ages, spectra,
                   wav_chunk=100):
    """
    Yield per-particle SEDs in wavelength chunks - memory-efficient spectral
    interpolation for IFU cube generation.

    Uses bilinear log(Z)-log(age) interpolation (same as
    :func:`assign_fluxes`) but returns the full SED rather than integrating
    through a filter.  Processing in wavelength chunks keeps peak memory at
    O(N * wav_chunk) rather than O(N * n_lam).

    Parameters
    ----------
    mass : array (n_p,)
        Initial stellar mass [Msun].
    metallicity, age : arrays (n_p,)
        Metallicity [absolute Z] and age [yr].
    metallicities : array (n_Z,)
        Template metallicity grid.
    ages : array (n_ages,)
        Template age grid.
    spectra : array (n_Z, n_ages, n_lam)
        Template SEDs [Lsun/Angstrom/Msun].
    wav_chunk : int
        Wavelength channels per iteration. Reduce if memory is running out.

    Yields
    ------
    l1, l2 : int
        Wavelength slice [l1 : l2] in the template wavelength grid.
    L_lam : array (n_p, l2 - l1)
        Per-particle luminosity [Lsun/Angstrom] for that slice.
    """
    mass        = np.asarray(mass,        dtype=np.float32).ravel()
    metallicity = np.asarray(metallicity, dtype=np.float64).ravel()
    age         = np.asarray(age,         dtype=np.float64).ravel()

    age_mask = ages > 0
    age_grid = ages[age_mask]
    spec     = spectra[:, age_mask, :]

    i_Z, i_Z1, i_a, i_a1, w_Z, w_a = _interp_weights(
        metallicity, age, metallicities, age_grid
    )
    wZ = w_Z[:, np.newaxis]
    wa = w_a[:, np.newaxis]
    m  = mass[:, np.newaxis]

    n_lam = spec.shape[2]
    for l1 in range(0, n_lam, wav_chunk):
        l2   = min(l1 + wav_chunk, n_lam)
        S00  = spec[i_Z,  i_a,  l1:l2]
        S10  = spec[i_Z1, i_a,  l1:l2]
        S01  = spec[i_Z,  i_a1, l1:l2]
        S11  = spec[i_Z1, i_a1, l1:l2]
        L_lam = m * ((1-wZ)*(1-wa)*S00 + wZ*(1-wa)*S10 +
                     (1-wZ)*  wa  *S01 + wZ*  wa  *S11)
        yield l1, l2, L_lam   # (n_p, chunk)  [Lsun/Angstrom]


def make_ifu_cube(X, X_c, mass, metallicity, age, z_fix,
                  velocity=None,
                  imf='kroupa', wav_range=None, wav_chunk=100,
                  extent=20, pixel_size=1.0, pixel_size_kpc=None,
                  projection='xy',
                  njobs=4, k=5, n_resample=250,
                  template_dir=None, cosmology='WMAP7',
                  dust_cube=None, dust_opacity_file=None,
                  verbose=True):
    """
    Build a mock IFU spectral datacube with adaptive SPH smoothing.

    The particle SEDs are interpolated from the SSP templates and smoothed
    onto a spatial grid using the same adaptive Gaussian kernel as
    :func:`make_image`.  Wavelength channels are processed in chunks so peak
    memory stays at O(N * wav_chunk + n_wav_chunk * n_pix^2) rather than
    O(N * n_lam).

    kNN smoothing lengths are computed **once** and reused for every
    wavelength chunk - the density step does not repeat per chunk.

    Parameters
    ----------
    X : array (3, n_p)
        Particle positions [kpc].
    X_c : array (3,)
        Galaxy centre [kpc].
    mass : array (n_p,)
        Initial stellar mass [Msun].
    metallicity, age : arrays (n_p,)
        Metallicity [absolute Z] and stellar age [yr].
    z_fix : float
        Source redshift.  Used for the output wavelength axis
        (wav_obs = wav_rest * (1 + z_fix)) and the distance-modulus flux
        scaling.  Also used to convert ``pixel_size`` (arcsec) to kpc unless
        ``pixel_size_kpc`` is provided.  When ``z_fix=0`` the distance-modulus
        rescaling is skipped and cube values remain at the 10-pc reference flux
        (absolute-magnitude flux scale); ``pixel_size_kpc`` must be provided in
        this case.
    velocity : array (n_p,), optional
        Line-of-sight velocity for each particle [km/s].  When provided, a
        relativistic Doppler shift is applied per particle before spatial
        smoothing, encoding kinematic information in the spectral axis.
        The Doppler-shift path uses trilinear (Z, age, wavelength)
        interpolation and requires O(N * wav_chunk) extra working memory;
        reduce ``wav_chunk`` if memory is tight.  Default: None (no shift).
    imf : {'kroupa', 'chabrier', 'salpeter'}
        Template: ``'kroupa'`` (E-MILES, default), ``'chabrier'`` or
        ``'salpeter'`` (BC03).
    wav_range : (float, float), optional
        Rest-frame wavelength range [Angstrom] to include.  Defaults to the full
        template coverage.  Narrowing this is the fastest way to cut runtime.
    wav_chunk : int
        Wavelength channels processed per iteration.  Tune for memory/speed.
    extent : float
        Image half-size [kpc].
    pixel_size : float
        Angular pixel scale [arcsec].  Converted to kpc via the angular
        diameter distance at ``z_fix``.  Ignored when ``pixel_size_kpc`` is
        provided.
    pixel_size_kpc : float, optional
        Physical pixel size [kpc].  When given, bypasses the arcsec→kpc
        conversion entirely, allowing ``z_fix=0`` and direct physical-unit
        control.  Takes precedence over ``pixel_size``.
    projection : {'xy', 'xz', 'yz'}
        Line-of-sight projection axis for spatial smoothing and, when a
        dust cube is provided, the direction of column-density integration.
    njobs : int
        Parallel workers for the smoothing step.
    nsteps : int
        Density bins for adaptive smoothing.
    k : int
        k-NN neighbours for density estimation.
    template_dir : str, optional
        Override default template directory.
    cosmology : str or astropy cosmology
        Cosmology for pixel scale and distance modulus.
    dust_cube : ndarray (n_pix, n_pix, n_pix), optional
        3D dust mass density [M_sun / kpc^3] from :func:`build_dust_cube`.
        When provided, a wavelength-dependent attenuation
        exp(-kappa(lambda) * Sigma_dust) is applied to each particle's SED before
        spatial smoothing.
    dust_opacity_file : str, optional
        Weingartner & Draine (2001) opacity table.  See
        :func:`apply_dust_attenuation` for format details.
    verbose : bool

    Returns
    -------
    cube : array (n_wav, n_pix, n_pix), float32
        Spectral datacube in AB-flux units per Angstrom, rescaled to the luminosity
        distance at ``z_fix``.  At ``z_fix=0`` values are at the 10-pc
        reference distance (absolute-magnitude flux scale).  The white-light
        collapse cube.sum(axis=0) * d_lambda should agree with :func:`make_image`
        output (before PSF and noise) for the same spatial resolution.
    wav_obs : array (n_wav,)
        Observer-frame wavelength axis [Angstrom] (axis 0 of cube).
    """
    import importlib; s3d = importlib.import_module('make_mocks.smooth_3d')

    cosmo = _resolve_cosmology(cosmology)
    if pixel_size_kpc is not None:
        res = float(pixel_size_kpc)
    else:
        if z_fix == 0.0:
            raise ValueError(
                "z_fix=0 makes arcsec→kpc conversion undefined.  "
                "Provide pixel_size_kpc instead."
            )
        res = pixel_size / cosmo.arcsec_per_kpc_proper(z_fix).value   # kpc/pixel

    X_c   = np.asarray(X_c)
    X_cen = X - X_c[:, np.newaxis]

    # Load and prepare templates
    wav, mets, ages, spec = load_templates(imf, template_dir)

    age_mask = ages > 0
    age_grid = ages[age_mask]
    spec     = spec[:, age_mask, :]

    if wav_range is not None:
        w_mask = (wav >= wav_range[0]) & (wav <= wav_range[1])
        if not w_mask.any():
            raise ValueError(
                f"wav_range {wav_range} Angstrom has no overlap with the template "
                f"wavelength grid ({wav[0]:.0f}-{wav[-1]:.0f} Angstrom)."
            )
        wav  = wav[w_mask]
        spec = spec[:, :, w_mask]

    n_lam   = len(wav)
    wav_obs = wav * (1.0 + z_fix)

    if verbose:
        print(f"    Template : {imf}  |  {n_lam} channels "
              f"({wav[0]:.0f}-{wav[-1]:.0f} Angstrom rest)  |  "
              f"{len(age_grid)} ages  |  {len(mets)} Z")

    # Validate particle inputs
    mass_p = np.asarray(mass,        dtype=np.float32).ravel()
    met_p  = np.asarray(metallicity, dtype=np.float64).ravel()
    age_p  = np.asarray(age,         dtype=np.float64).ravel()

    bad_Z = ~np.isfinite(met_p) | (met_p <= 0)
    if bad_Z.any():
        warnings.warn(
            f"[WARNING] make_ifu_cube: {bad_Z.sum()} particles with non-positive Z. "
            f"Clipped to Z_min = {mets[0]:.4g}", stacklevel=2
        )
        met_p = np.where(bad_Z, mets[0], met_p)

    bad_a = ~np.isfinite(age_p) | (age_p <= 0)
    if bad_a.any():
        warnings.warn(
            f"[WARNING] make_ifu_cube: {bad_a.sum()} particles with non-positive age. "
            f"Clipped to age_min = {age_grid[0]:.3e} yr. No time travelers allowed.", stacklevel=2
        )
        age_p = np.where(bad_a, age_grid[0], age_p)

    n_p = len(mass_p)

    # Per-particle Doppler factors (optional)
    _C_KMS = 299792.458
    if velocity is not None:
        vel_p   = np.asarray(velocity, dtype=np.float64).ravel()
        doppler = np.sqrt((1.0 + vel_p / _C_KMS) / (1.0 - vel_p / _C_KMS))  # (n_p,)
        dz      = doppler * (1.0 + z_fix)   # (n_p,) combined redshift factor
    else:
        dz = None

    # Compute kNN distances ONCE
    if verbose:
        print("[START] Computing kNN smoothing lengths")
    density, dist = s3d.nearest_neighbour_density(X_cen.T, k=k, njobs=njobs)

    # Pre-scatter particle positions once so every wavelength chunk shares the
    # same spatial PSF (consistent smoothing across all spectral channels).
    if k is not None and k > 0:
        if verbose:
            print(f"[START] Pre-scattering {n_p} particles "
                  f"({n_resample} resamples each) ...")
        smooth_state = s3d.precompute_smoothing_state(
            X_cen, res=res, extent=extent, projection=projection,
            k=k, njobs=njobs,
            precomputed_dist=dist, precomputed_density=density,
            n_resample=n_resample,
        )
    else:
        smooth_state = None

    # Precompute bilinear weights (same for all wavelength chunks)
    i_Z, i_Z1, i_a, i_a1, w_Z, w_a = _interp_weights(met_p, age_p, mets, age_grid)
    wZ = w_Z[:, np.newaxis]
    wa = w_a[:, np.newaxis]
    m  = mass_p[:, np.newaxis]

    # At z=0, d_L=0 so distmod=-inf; keep fluxes at the 10-pc reference.
    dist_scale = (10.0 ** (-cosmo.distmod(z_fix).value / 2.5)
                  if z_fix > 0.0 else 1.0)

    # Dust attenuation setup (optional)
    if dust_cube is not None:
        n_pix_d  = dust_cube.shape[0]
        dz_d     = 2.0 * extent / n_pix_d
        los_ax   = {'xy': 2, 'xz': 1, 'yz': 0}[projection]
        col_total = dust_cube.sum(axis=los_ax, keepdims=True) * dz_d
        col_gcm2  = ((col_total - np.cumsum(dust_cube, axis=los_ax) * dz_d)
                     * (_MSUN_G / _KPC_CM ** 2))   # g / cm^2, dust from particle to observer

        edges_d = np.linspace(-extent, extent, n_pix_d + 1)
        def _idx_d(arr):
            return np.clip(np.digitize(arr, edges_d) - 1, 0, n_pix_d - 1)

        sigma_gcm2 = col_gcm2[_idx_d(X_cen[0]),
                               _idx_d(X_cen[1]),
                               _idx_d(X_cen[2])]   # (n_p,)

        _opac_file = dust_opacity_file or _DUST_OPACITY_FILE
        _f = np.loadtxt(_opac_file, skiprows=50)
        # Pre-interpolate kappa at every rest-frame template wavelength [cm^2/g]
        kappa_wav = np.interp(wav, _f[:, 0] * 1e4, _f[:, 4]).astype(np.float32)
        del _f
        if verbose:
            print("[INFO] Dust attenuation enabled - kappa(lambda) applied per channel")
    else:
        sigma_gcm2 = None
        kappa_wav  = None

    # Allocate cube
    n_pix = int(extent * 2.0 / res) - 1
    cube  = np.zeros((n_lam, n_pix, n_pix), dtype=np.float32)

    n_chunks = (n_lam + wav_chunk - 1) // wav_chunk
    if verbose:
        print(f"[START] Building {n_pix}x{n_pix}x{n_lam} cube in {n_chunks} chunks ...")

    try:
        import tqdm as _tqdm
        _pbar = _tqdm.tqdm(total=n_chunks, desc='    wav chunks') if verbose else None
    except ImportError:
        _pbar = None

    # Wavelength chunk loop
    for l1 in range(0, n_lam, wav_chunk):
        l2 = min(l1 + wav_chunk, n_lam)

        if dz is None:
            # --- No Doppler shift: bilinear (Z, age) interpolation ---
            sl    = spec[:, :, l1:l2]
            L_lam = m * (
                (1-wZ)*(1-wa)*sl[i_Z,  i_a,  :] + wZ*(1-wa)*sl[i_Z1, i_a,  :] +
                (1-wZ)*  wa  *sl[i_Z,  i_a1, :] + wZ*  wa  *sl[i_Z1, i_a1, :]
            )

            # Wavelength-dependent dust attenuation
            if sigma_gcm2 is not None:
                tau    = kappa_wav[l1:l2][np.newaxis, :] * sigma_gcm2[:, np.newaxis]
                L_lam *= np.exp(-tau)

            # Convert to AB flux units at the luminosity distance
            # shape (chunk, n_p) for smooth_3d's quantity_sum convention
            fluxes_chunk = (L_lam * (_C10PC_AB * dist_scale)).T.astype(np.float32)

        else:
            # --- Per-particle Doppler shift: trilinear (Z, age, wavelength) ---
            # Output observed wavelengths for this chunk (cosmological redshift only)
            lam_obs_chunk = wav_obs[l1:l2]                      # (chunk,)

            # Per-particle rest-frame wavelengths for each output channel
            # lam_rest[i, j] = rest-frame wavelength of particle i that maps to
            #                  the observed wavelength lam_obs_chunk[j]
            lam_rest = lam_obs_chunk[np.newaxis, :] / dz[:, np.newaxis]  # (n_p, chunk)

            # Find wavelength bracket indices in the template grid
            n_chunk = l2 - l1
            flat_lr = lam_rest.ravel()
            i_lam   = np.searchsorted(wav, flat_lr) - 1
            i_lam   = np.clip(i_lam, 0, n_lam - 2).reshape(n_p, n_chunk)
            i_lam1  = i_lam + 1
            w_lam   = ((lam_rest - wav[i_lam]) /
                       (wav[i_lam1] - wav[i_lam]))               # (n_p, chunk)

            # Trilinear interpolation: bilinear in (Z, age), linear in wavelength
            def _S(iZ_arr, ia_arr):
                lo = spec[iZ_arr[:, np.newaxis], ia_arr[:, np.newaxis], i_lam ]
                hi = spec[iZ_arr[:, np.newaxis], ia_arr[:, np.newaxis], i_lam1]
                return lo * (1.0 - w_lam) + hi * w_lam           # (n_p, chunk)

            L_lam = m * (
                (1-wZ)*(1-wa)*_S(i_Z,  i_a ) +
                   wZ *(1-wa)*_S(i_Z1, i_a ) +
                (1-wZ)*   wa *_S(i_Z,  i_a1) +
                   wZ *   wa *_S(i_Z1, i_a1)
            )  # (n_p, chunk) [Lsun/Angstrom at each particle's rest-frame wavelength]

            # F_lambda_obs = L_lambda(lam_rest) / dz^2 * flux_scale
            # (one dz factor for wavelength stretch, one for energy dimming)
            fluxes_chunk = (
                L_lam / dz[:, np.newaxis] ** 2 * (_C10PC_AB * dist_scale)
            ).T.astype(np.float32)  # (chunk, n_p)

        # Smooth onto spatial grid
        if smooth_state is not None:
            imgs = s3d.smooth_with_state(smooth_state, fluxes_chunk)
        else:
            # k=0 or None: no smoothing, simple histogram
            if verbose and l1 == 0:
                print("[INFO] k=0 or None: no smoothing, using simple 2D histogram for each channel.")
            imgs = [s3d.bin_particles(X_cen, fluxes_chunk[i], projection=projection,
                                      res=res, extent=extent)
                    for i in range(fluxes_chunk.shape[0])]

        for di, img in enumerate(imgs):
            cube[l1 + di] = img

        if _pbar is not None:
            _pbar.update(1)

    if _pbar is not None:
        _pbar.close()

    if verbose:
        msg = f"[END] make_ifu_cube: Created IFU cube with shape {cube.shape} and wavelength range {wav_obs[0]:.1f}-{wav_obs[-1]:.1f} (n_wav={len(wav_obs)})."
        print(msg)
    return cube, wav_obs


# Backward-compatible alias (will remove)

def wrapper(X, X_c, mass, metallicity, age, mu_lims=None, psfs=None,
            projection='xy', z_fix=0.1, imf='chab', extent=20,
            pixel_size=0.2, njobs=4, nsteps=500, n_resample=500, k=5,
            bands=None, cosmology='WMAP7', verbose=True):
    """Backward-compatible alias for :func:`make_image`.

    .. note::
        Metallicity convention change: this function now expects metallicity
        as an **absolute mass fraction** (solar approx 0.02), the same as
        :func:`make_image`.  Code previously normalised to solar units
        (e.g. ``met / 0.02``).
    """
    _imf_map = {
        'chab': 'chabrier', 'chabrier': 'chabrier',
        'salp': 'salpeter', 'salpeter': 'salpeter',
        'krou': 'kroupa',   'kroupa':   'kroupa',
    }
    imf_resolved = _imf_map.get(imf.lower(), imf)
    if bands is None:
        bands = ['g', 'r', 'i']
    return make_image(
        X, X_c, mass, metallicity, age, mu_lims, psfs,
        projection=projection, z_fix=z_fix, imf=imf_resolved, extent=extent,
        pixel_size=pixel_size, njobs=njobs, nsteps=nsteps,
        n_resample=n_resample, k=k, bands=bands,
        cosmology=cosmology, verbose=verbose,
    )
