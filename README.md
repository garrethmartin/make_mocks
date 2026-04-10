
# make_mocks

**make_mocks** is a Python package for generating mock galaxy images and IFU cubes.

---

## Features

- Generate mock galaxy images in arbitrary filters (Euclid, LSST, Roman, HST, etc.)
- Adaptive smoothing of particle data (SPH-like, kNN-based)
- PSF convolution and noise
- Dust attenuation

---

## Installation

From the `make_mocks` directory, install with:

```bash
pip install .
```

Dependencies: numpy, scipy, astropy, h5py, (optional: pykdtree for speed)

---


## Quickstart Example

```python
import make_mocks as mm

# Example: create a 3-band image from particle data
imgs_noisy, imgs_clean = mm.make_image(
    X, X_c, mass, metallicity, age,
    mu_lims=[29.5, 30.0, 30.5],
    psfs=[psf_g, psf_r, psf_i],
    bands=['g', 'r', 'i'],
    z_fix=0.1,
)
# imgs_noisy is None if mu_lims is not given; imgs_clean always holds noiseless images.
```

---

## Interactive Demo

For a step-by-step demonstration, see the Jupyter notebook:

- **demo_make_mocks.ipynb** — shows example usage and visualization of outputs.

---

## API Overview

### Main Functions

- `make_image(...)`: Full pipeline for mock image generation (see below)
- `calculate_fluxes(...)`: Compute per-particle fluxes in AB units
- `assign_fluxes(...)`: Interpolate template grid to assign band luminosities
- `precompute_band_fluxes(...)`: Integrate SEDs through multiple filters
- `integrate_band(...)`: Integrate SED through a single filter
- `build_dust_cube(...)`: Build a 3D dust density cube from gas particles
- `apply_dust_attenuation(...)`: Attenuate fluxes using a dust cube
- `smooth_3d.smooth_3d(...)`: Adaptive 2D smoothing of particles
- `smooth_3d.bin_particles(...)`: Simple 2D histogram (no smoothing)

See the docstrings in `make_mocks.py` and `smooth_3d.py` for full parameter details and advanced options.

---

## Templates and Filters

Stellar population templates and filter curves are stored in the `templates/` directory:

- **Templates:**
	- `bc03_chabrier.hdf5`, `bc03_salpeter.hdf5`, `emiles_kroupa.hdf5` (HDF5 format)
- **PSFs:**
	- e.g. `psf_g-band.fits`, `psf_VIS.fits`, etc.
- **Filters:**
	- In `templates/filters/` (e.g. `g`, `r`, `VIS`, `F062`, ...)
- **Dust opacity:**
	- `kext_albedo_WD_MW_3.1_60.txt` (Weingartner & Draine MW dust)

You can add your own templates, PSFs, or filters by placing files in these folders and referencing them by name.

---

## Example: Full Pipeline

```python
import make_mocks as mm

# Assume you have arrays: X, X_c, mass, metallicity, age, psfs
imgs_noisy, imgs_clean = mm.make_image(
		X, X_c, mass, metallicity, age,
		mu_lims=[29.5, 30.0, 30.5],
		psfs=[psf_g, psf_r, psf_i],
		bands=['g', 'r', 'i'],
		z_fix=0.1,
		imf='chabrier',
		extent=20,
		pixel_size=0.2,
		njobs=4,
		nsteps=500,
		k=5,
		template_dir='templates',
		filter_dir='templates/filters',
		cosmology='WMAP7',
		verbose=True,
)

# imgs_noisy: list of 2D arrays (one per band) with noise
# imgs_clean: list of 2D arrays (one per band) noiseless
```

---

## Notes

- **Output units:** All fluxes and images are in AB flux density units. To convert to observed AB magnitudes, use $m_\mathrm{AB} = -2.5 \log_{10}(f), where $f$ is the flux value.
- **Redshifting:** SEDs are shifted to the observer frame before filter integration.
- **Reference distance:** Fluxes are computed at 10 pc, then rescaled to the luminosity distance.
- **Metallicity:** Use absolute mass fraction (e.g. solar $Z_\odot \approx 0.02$).
- **IMF/template options:** `'chabrier'` (default), `'salpeter'`, `'kroupa'`.
