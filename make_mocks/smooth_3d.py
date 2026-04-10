"""
Adaptive particle smoothing onto 2-D projected images and 3-D density cubes.

Method
--------------
Each particle is assigned a smoothing length equal to the distance to its
k-th nearest neighbor in 3-D.  Smoothed particles are scattered onto the
target grid by drawing Gaussian offsets per particle, using the local kNN
distance as the smoothing scale.

This implements the adaptive smoothing approach of Merritt et al. 2020
(https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4570M). We use resampling
rather than direct convolution for better efficiency with large image sizes.
This approach is therefore an approximation which improves as "n_resample"
is increased.

Public API
----------
smooth_3d                  Adaptively smooth particles onto a projected 2-D image.
precompute_smoothing_state Pre-compute kNN state for repeated smoothing calls.
smooth_with_state          Apply pre-computed state to a new quantity array.
build_density_cube         Adaptively smooth particles onto a 3-D density cube.
nearest_neighbour_density  k-NN density estimate (used internally and externally).
bin_particles              Simple 2-D histogram without smoothing.

Parallelism
-----------
Particle batches are distributed across worker processes.  Each worker returns
partial image arrays; the main process sums them.  There is no shared mutable
state, so scaling with CPU count is near-linear (hopefully).
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from fast_histogram import histogram2d

_GAUSSIAN_TRUNCATE = 3.0

__all__ = [
    'smooth_3d',
    'precompute_smoothing_state',
    'smooth_with_state',
    'build_density_cube',
    'nearest_neighbour_density',
    'bin_particles',
    'project',
    'img_extent',
]


# Coordinate helpers

def project(X, projection='xy'):
    """Extract two projected coordinate arrays from a (3, N) position array.

    Parameters
    ----------
    X : array, shape (3, N)
        3-D particle positions.
    projection : {'xy', 'xz', 'yz'}
        Axis pair to project onto.

    Returns
    -------
    x, y : tuple of 1-D arrays
    """
    axes = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    if projection not in axes:
        raise ValueError(f"projection must be 'xy', 'xz', or 'yz', got {projection!r}")
    a, b = axes[projection]
    return X[a], X[b]


def img_extent(x, y):
    """Smallest integer that encompasses all (x, y) coordinate values."""
    return int(max(np.max(np.abs(x)), np.max(np.abs(y))))


# ---------------------------------------------------------------------------
def _gaussian_smooth_2d(arr, sigma):
    """Gaussian-smooth a 2-D array."""
    return gaussian_filter(arr, sigma, truncate=_GAUSSIAN_TRUNCATE)


# Density estimation

def nearest_neighbour_density(X, k=5, njobs=1, verbose=False):
    """k-nearest-neighbor density estimate in 3-D.

    Uses the sphere enclosing the k-th true neighbor (self excluded):

        rho_i = k / [(4*pi/3) d_{k,i}^3]

    Backend priority:

    1. ``pykdtree`` (optional) - pure C++, fastest for float32 and small k.
       Install with: ``pip install pykdtree``
    2. ``scipy.spatial.cKDTree`` - always available; queries run in parallel
       across ``njobs`` threads (pass ``njobs=-1`` to use all cores).

    Parameters
    ----------
    X : array, shape (N, 3)
        Particle positions.
    k : int
        Number of nearest neighbors.
    njobs : int
        Number of query threads (scipy backend only).

    Returns
    -------
    density : array, shape (N,)
        Number density [1 / (units of X)^3].
    dist : array, shape (N,)
        Distance to the k-th nearest neighbor.
    """
    # Query k+1 neighbours: index 0 is the point itself (distance 0),
    # indices 1..k are the k true neighbours.
    try:
        from pykdtree.kdtree import KDTree as _PyKDTree
        tree = _PyKDTree(X.astype(np.float32), leafsize=16)
        dist, _ = tree.query(X.astype(np.float32), k=k + 1)
        dist_k = dist[:, k].astype(np.float64)
    except ImportError:
        from scipy.spatial import cKDTree
        tree = cKDTree(X)
        if njobs > 1 and verbose:
            print(f"[INFO] Using {njobs} threads for kNN search.")
        dist, _ = tree.query(X, k=k + 1, workers=njobs)
        dist_k = dist[:, k]

    volume = (4.0 / 3.0) * np.pi * dist_k ** 3
    return k / volume, dist_k


# Binned projection (utility, no smoothing)

def bin_particles(X, quantity, projection='xy', res=1.0, statistic='sum',
                  extent=None):
    """Simple 2-D histogram of a particle quantity (no smoothing).

    Parameters
    ----------
    X : array, shape (3, N)
        Particle positions.
    quantity : array, shape (N,)
        Scalar quantity to bin.
    projection : {'xy', 'xz', 'yz'}
    res : float
        Pixel size in the same units as X.
    statistic : str
        Passed to ``scipy.stats.binned_statistic_2d``.
    extent : float, optional
        Image half-width.  Defaults to the maximum |coordinate| value.

    Returns
    -------
    h : 2-D array
    """
    from scipy.stats import binned_statistic_2d
    x, y = project(X, projection=projection)
    if extent is None:
        extent = img_extent(x, y)
    bins = np.linspace(-extent, extent, int(extent * 2.0 / res))
    h, _, _, _ = binned_statistic_2d(x, y, quantity, statistic=statistic,
                                     bins=bins)
    return h


# Multiprocessing worker

# Module-level dict populated before pool creation. Fork: copy-on-write inherited. Spawn: set via pool initializer.
_worker_context: dict = {}


def _worker_init(x_s, y_s, q_sum_s, q_avg_s, res=1.0, dist_s=None):
    """Pool initializer: store sorted particle arrays in this worker's context."""
    global _worker_context
    _worker_context = {'x': x_s, 'y': y_s, 'qs': q_sum_s, 'qa': q_avg_s,
                       'res': res, 'dist': dist_s}


def _scatter_worker(args):
    """Scatter-smooth a batch of particles [b0:b1] using per-particle smoothing lengths.

    Each particle is scattered ``n_resample`` times with a 2-D Gaussian of std
    equal to its kNN distance (the natural smoothing length in position units).

    Parameters (packed into args)
    ------------------------------
    b0, b1       : particle slice into the sorted arrays in _worker_context
    extent       : image half-width
    n_pix        : image size in pixels
    n_sum, n_avg : number of sum/average channels
    n_resample   : duplicates per particle
    """
    import numpy as np
    from fast_histogram import histogram2d as _h2d

    b0, b1, extent, n_pix, n_sum, n_avg, n_resample = args
    x_s    = _worker_context['x']
    y_s    = _worker_context['y']
    dist_s = _worker_context['dist']
    q_sum_s = _worker_context['qs']
    q_avg_s = _worker_context['qa']
    range2d = [[-extent, extent], [-extent, extent]]

    bm  = b1 - b0
    tot = bm * n_resample

    rng = np.random.default_rng()
    sigma_rep = np.repeat(dist_s[b0:b1].astype(np.float32), n_resample)
    dx  = rng.standard_normal(tot).astype(np.float32) * sigma_rep
    dy  = rng.standard_normal(tot).astype(np.float32) * sigma_rep
    xb  = np.repeat(x_s[b0:b1].astype(np.float32), n_resample) + dx
    yb  = np.repeat(y_s[b0:b1].astype(np.float32), n_resample) + dy

    out_sum     = np.zeros((n_sum, n_pix, n_pix), dtype=np.float32) if n_sum > 0 else None
    out_avg_num = np.zeros((n_avg, n_pix, n_pix), dtype=np.float32) if n_avg > 0 else None
    out_avg_den = np.zeros((n_avg, n_pix, n_pix), dtype=np.float32) if n_avg > 0 else None

    if n_sum > 0:
        for i in range(n_sum):
            w = np.repeat(q_sum_s[i, b0:b1].astype(np.float32), n_resample) / n_resample
            out_sum[i] = _h2d(xb, yb, weights=w, bins=n_pix, range=range2d)

    if n_avg > 0:
        cnt   = np.ones(tot, dtype=np.float32) / n_resample
        h_cnt = _h2d(xb, yb, weights=cnt, bins=n_pix, range=range2d).astype(np.float32)
        for i in range(n_avg):
            w = np.repeat(q_avg_s[i, b0:b1].astype(np.float32), n_resample) / n_resample
            out_avg_num[i] = _h2d(xb, yb, weights=w, bins=n_pix, range=range2d)
        out_avg_den += h_cnt[np.newaxis]

    return out_sum, out_avg_num, out_avg_den


def _run_scatter(x_s, y_s, dist_s, q_sum_s, q_avg_s,
                 extent, n_pix, n_sum, n_avg, n_resample,
                 scatter_batch, njobs, verbose):
    """Scatter all smoothed particles using their individual kNN distances.

    Each particle is duplicated ``n_resample`` times with Gaussian offsets
    scaled by its own kNN distance.  Particles are processed in batches of
    ``scatter_batch``; batches are distributed across ``njobs`` workers.

    Returns
    -------
    out_sum, out_avg_num, out_avg_den : arrays or None
    """
    out_sum     = np.zeros((n_sum, n_pix, n_pix), dtype=np.float32) if n_sum > 0 else None
    out_avg_num = np.zeros((n_avg, n_pix, n_pix), dtype=np.float32) if n_avg > 0 else None
    out_avg_den = np.zeros((n_avg, n_pix, n_pix), dtype=np.float32) if n_avg > 0 else None

    def _reduce(s, an, ad):
        if s  is not None: out_sum[:]     += s
        if an is not None: out_avg_num[:] += an; out_avg_den[:] += ad

    N      = len(x_s)
    ranges = [(b0, min(b0 + scatter_batch, N))
              for b0 in range(0, N, scatter_batch)]
    worker_args = [(b0, b1, extent, n_pix, n_sum, n_avg, n_resample)
                   for b0, b1 in ranges]

    if verbose:
        n_batches = len(ranges)
        avg_ppp   = N / n_batches if n_batches > 0 else 0
        print(f"[INFO] Scatter: {N} particles, {n_batches} batch(es) of "
              f"≤{scatter_batch} particles, {n_resample} resamples/particle "
              f"({N * n_resample:,} total sub-particles).")

    if njobs > 1:
        from multiprocessing import Pool
        with Pool(processes=njobs, initializer=_worker_init,
                  initargs=(x_s, y_s, q_sum_s, q_avg_s, 1.0, dist_s)) as pool:
            if verbose:
                print(f"[INFO] Using {njobs} parallel workers.")
            iterator = pool.imap_unordered(_scatter_worker, worker_args)
            if verbose:
                try:
                    import tqdm
                    iterator = tqdm.tqdm(iterator, total=len(worker_args))
                except ImportError:
                    pass
            for result in iterator:
                _reduce(*result)
    else:
        global _worker_context
        _worker_context = {'x': x_s, 'y': y_s, 'qs': q_sum_s, 'qa': q_avg_s,
                           'res': 1.0, 'dist': dist_s}
        _iter = worker_args
        if verbose:
            try:
                import tqdm
                _iter = tqdm.tqdm(_iter, total=len(worker_args))
            except ImportError:
                pass
        for a in _iter:
            _reduce(*_scatter_worker(a))

    return out_sum, out_avg_num, out_avg_den


# Public API: 2-D projection

def smooth_3d(
    X,
    quantity_sum=None,
    quantity_average=None,
    res=1.0,
    extent=None,
    upper_threshold=None,
    lower_threshold=None,
    njobs=1,
    nsteps=None,
    k=5,
    n_resample=500,
    scatter_batch=50_000,
    projection='xy',
    verbose=True,
    antialias=True,
    precomputed_dist=None,
    precomputed_density=None,
    sigma_step=0.5,
    sigma_min=0.5,
    method='scatter',
):
    """Adaptively smooth particles onto a projected 2-D image.

    Sparse particles are spread broadly; dense particles are spread narrowly,
    according to the local k-nearest-neighbour density.

    No sigma grouping or convolution is used anymore, but sigma_step and 
    sigma_min are still included for now (but now ignored).

    Parameters
    ----------
    X : array, shape (3, N)
        3-D particle positions in any consistent length unit.
    quantity_sum : array, shape (Q, N), optional
        Quantities whose projected sum is required (e.g. flux, mass).
    quantity_average : array, shape (Q, N), optional
        Quantities whose projected density-weighted average is required.
    res : float
        Pixel size in the same units as X.
    extent : float, optional
        Image half-width; the image spans [-extent, +extent] in both axes.
        Defaults to the maximum |coordinate| in the projected plane.
    upper_threshold : float, optional
        Particles with number density (in units of res^-3) above this value
        are binned directly without smoothing.
    lower_threshold : float, optional
        Particles below this density threshold are excluded entirely.
    njobs : int
        Number of parallel worker processes.
    nsteps : int, optional
        Deprecated and ignored.
    k : int
        Number of nearest neighbours for the density estimate.
    n_resample : int
        Number of sub-particles per particle for scatter smoothing.
        Default 500.
    scatter_batch : int
        Maximum number of particles processed per mini-batch in the scatter
        method.  Controls peak memory usage.  Default 50_000.
    projection : {'xy', 'xz', 'yz'}
        Projection axis pair.
    verbose : bool
    antialias : bool
        If True, apply a 1-pixel Gaussian to suppress pixel-scale aliasing.
    precomputed_dist : array, shape (N,), optional
        Pre-computed kNN distances.  If provided together with
        ``precomputed_density``, the kNN step is skipped.
    precomputed_density : array, shape (N,), optional
        Pre-computed number densities matching ``precomputed_dist``.
    sigma_step : float, optional
        Deprecated and ignored.
    sigma_min : float, optional
        Deprecated and ignored.
    method : {'scatter'}
        Only scatter smoothing is supported.

    Returns
    -------
    img : list of 2-D arrays (one per quantity_sum band), or np.nan
    average_img : list of 2-D arrays (one per quantity_average band), or np.nan
    """
    import warnings

    # Backward-compat: old API used [False] sentinels
    if isinstance(quantity_sum,     list) and quantity_sum     == [False]:
        quantity_sum = None
    if isinstance(quantity_average, list) and quantity_average == [False]:
        quantity_average = None

    if nsteps is not None:
        warnings.warn(
            "nsteps is deprecated and ignored.",
            DeprecationWarning, stacklevel=2,
        )

    if sigma_step != 0.5 or sigma_min != 0.5:
        warnings.warn(
            "sigma_step and sigma_min are deprecated and ignored in scatter-only mode.",
            DeprecationWarning, stacklevel=2,
        )

    if method != 'scatter':
        raise ValueError("Only method='scatter' is supported.")

    if quantity_sum is None and quantity_average is None:
        raise ValueError(
            "[WARNING] At least one of quantity_sum or quantity_average must be provided."
        )

    x, y = project(X, projection)
    if extent is None:
        extent = img_extent(x, y)
    n_pix   = int(extent * 2.0 / res) - 1
    range2d = [[-extent, extent], [-extent, extent]]

    # If k=0 or None, do not smooth, just bin particles
    if k is None or k == 0:
        if verbose:
            print("[INFO] k=0 or None: no smoothing, using simple 2D histogram.")
        img = []
        if quantity_sum is not None:
            for i in range(quantity_sum.shape[0]):
                h = bin_particles(X, quantity_sum[i], projection=projection, res=res, extent=extent)
                img.append(h)
        else:
            img = np.nan
        average_img = []
        if quantity_average is not None:
            for i in range(quantity_average.shape[0]):
                h = bin_particles(X, quantity_average[i], projection=projection, res=res, extent=extent)
                average_img.append(h)
        else:
            average_img = np.nan
        return img, average_img

    # ...existing code for smoothing...
    # 1. kNN density
    if precomputed_dist is not None and precomputed_density is not None:
        dist    = np.asarray(precomputed_dist)
        density = np.asarray(precomputed_density)
    else:
        if verbose:
            print("[START] Computing local particle density (kNN).")
        density, dist = nearest_neighbour_density(X.T, k=k, njobs=njobs, verbose=verbose)

    # 2. Density threshold mask
    d_element = density / res ** 3
    mask = np.ones(len(d_element), dtype=bool)
    if upper_threshold is not None:
        mask &= d_element < upper_threshold
    if lower_threshold is not None:
        mask &= d_element > lower_threshold

    # 3. Sort smoothed particles by kNN distance (sparse to dense)
    sort_idx = np.argsort(dist[mask])
    x_s      = x[mask][sort_idx]
    y_s      = y[mask][sort_idx]
    dist_s   = dist[mask][sort_idx]

    n_sum = quantity_sum.shape[0]     if quantity_sum     is not None else 0
    n_avg = quantity_average.shape[0] if quantity_average is not None else 0
    q_sum_s = quantity_sum    [:, mask][:, sort_idx] if n_sum > 0 else None
    q_avg_s = quantity_average[:, mask][:, sort_idx] if n_avg > 0 else None

    # 4. Smooth (scatter-only)
    if verbose:
        print("[INFO] Smoothing particle distribution.")
    out_sum, out_avg_num, out_avg_den = _run_scatter(
        x_s, y_s, dist_s, q_sum_s, q_avg_s,
        extent, n_pix, n_sum, n_avg, n_resample, scatter_batch, njobs, verbose,
    )

    # 5. Add high-density (unsmoothed) particles and apply anti-alias
    sigma_aa     = 1.0 if antialias else 0.0
    n_unsmoothed = int(np.count_nonzero(~mask))

    img = []
    if n_sum > 0:
        for i in range(n_sum):
            h = out_sum[i].copy()
            if n_unsmoothed > 0:
                h += histogram2d(x[~mask], y[~mask],
                                 weights=quantity_sum[i][~mask],
                                 bins=n_pix, range=range2d)
            h[~np.isfinite(h)] = 0.0
            img.append(_gaussian_smooth_2d(h, sigma_aa) if sigma_aa > 0 else h)
    else:
        img = np.nan

    average_img = []
    if n_avg > 0:
        for i in range(n_avg):
            num = out_avg_num[i].copy()
            den = out_avg_den[i].copy()
            if n_unsmoothed > 0:
                num += histogram2d(x[~mask], y[~mask],
                                   weights=quantity_average[i][~mask],
                                   bins=n_pix, range=range2d)
                den += histogram2d(x[~mask], y[~mask],
                                   bins=n_pix, range=range2d)
            num[~np.isfinite(num)] = 0.0
            den[~np.isfinite(den)] = 0.0
            with np.errstate(invalid='ignore', divide='ignore'):
                avg = np.where(den > 0.0, num / den, 0.0)
            average_img.append(_gaussian_smooth_2d(avg, sigma_aa) if sigma_aa > 0 else avg)
    else:
        average_img = np.nan

    return img, average_img


def precompute_smoothing_state(X, res, extent=None, projection='xy',
                                k=5, njobs=1, nsteps=None,
                                upper_threshold=None, lower_threshold=None,
                                sigma_step=0.5, sigma_min=0.5,
                                n_resample=0,
                                precomputed_dist=None,
                                precomputed_density=None):
    """Pre-compute kNN density and bin structure for repeated smoothing.

    Call this once for a set of particle positions, then pass the returned
    state to :func:`smooth_with_state` for each quantity to smooth (e.g. each
    wavelength chunk of a spectral cube).  The expensive kNN step runs once.

    Parameters
    ----------
    X : array, shape (3, N)
        Particle positions.
    res : float
        Pixel size (same units as X).
    extent : float, optional
        Image half-width.  Defaults to the maximum |coordinate| value.
    projection : {'xy', 'xz', 'yz'}
    k : int
        Number of nearest neighbours for the density estimate.
    njobs : int
        Parallel threads for the kNN query.
    nsteps : int, optional
        Deprecated and ignored.
    upper_threshold : float, optional
        Particles above this number density (res⁻³ units) are unsmoothed.
    lower_threshold : float, optional
        Particles below this density are excluded.
    sigma_step : float, optional
        Deprecated and ignored.
    sigma_min : float, optional
        Deprecated and ignored.

    Returns
    -------
    state : dict
        Opaque dict for :func:`smooth_with_state`.
    """
    import warnings
    if nsteps is not None:
        warnings.warn(
            "nsteps is deprecated and ignored.",
            DeprecationWarning, stacklevel=2,
        )

    if sigma_step != 0.5 or sigma_min != 0.5:
        warnings.warn(
            "sigma_step and sigma_min are deprecated and ignored in scatter-only mode.",
            DeprecationWarning, stacklevel=2,
        )

    xa, ya = project(X, projection)
    if extent is None:
        extent = img_extent(xa, ya)
    n_pix   = int(extent * 2.0 / res) - 1
    range2d = [[-extent, extent], [-extent, extent]]

    if precomputed_dist is not None and precomputed_density is not None:
        dist    = np.asarray(precomputed_dist)
        density = np.asarray(precomputed_density)
    else:
        density, dist = nearest_neighbour_density(X.T, k=k, njobs=njobs)

    d_element = density / res ** 3
    mask = np.ones(len(d_element), dtype=bool)
    if upper_threshold is not None:
        mask &= d_element < upper_threshold
    if lower_threshold is not None:
        mask &= d_element > lower_threshold

    sort_idx = np.argsort(dist[mask])
    x_s      = xa[mask][sort_idx]
    y_s      = ya[mask][sort_idx]
    dist_s   = dist[mask][sort_idx]

    state = {
        'x_s':          x_s,
        'y_s':          y_s,
        'dist_s':       dist_s,
        'x_hi':         xa[~mask],
        'y_hi':         ya[~mask],
        'mask':         mask,
        'sort_idx':     sort_idx,
        'extent':       extent,
        'res':          res,
        'n_pix':        n_pix,
        'range2d':      range2d,
        'projection':   projection,
        'n_resample_sc': 0,
    }

    if n_resample > 0:
        rng       = np.random.default_rng()
        N_s       = len(x_s)
        sigma_rep = np.repeat(dist_s.astype(np.float32), n_resample)
        dx = rng.standard_normal(N_s * n_resample).astype(np.float32) * sigma_rep
        dy = rng.standard_normal(N_s * n_resample).astype(np.float32) * sigma_rep
        state['x_sc']         = np.repeat(x_s.astype(np.float32), n_resample) + dx
        state['y_sc']         = np.repeat(y_s.astype(np.float32), n_resample) + dy
        state['n_resample_sc'] = n_resample

    return state


def smooth_with_state(state, quantity_sum, njobs=1, antialias=False,
                      method='scatter', n_resample=100, scatter_batch=1000):
    """Smooth particles using a pre-computed smoothing state.

    Parameters
    ----------
    state : dict
        Returned by :func:`precompute_smoothing_state`.
    quantity_sum : array, shape (Q, N)
        Quantities to smooth.  N must match the original particle count.
    njobs : int
    antialias : bool
        Apply a 1-pixel Gaussian to each output image.
    method : {'scatter'}
        Only scatter smoothing is supported.
    n_resample : int
        Sub-particles per particle for the scatter method (default 100).
    scatter_batch : int
        Mini-batch size for the scatter method (default 1000).

    Returns
    -------
    imgs : list of 2-D arrays, length Q
        One smoothed image per quantity band.
    """
    x_s          = state['x_s']
    y_s          = state['y_s']
    dist_s       = state['dist_s']
    mask         = state['mask']
    sort_idx     = state['sort_idx']
    extent       = state['extent']
    res          = state['res']
    n_pix        = state['n_pix']
    range2d      = state['range2d']

    if method != 'scatter':
        raise ValueError("Only method='scatter' is supported.")

    quantity_sum = np.asarray(quantity_sum)
    if quantity_sum.ndim == 1:
        quantity_sum = quantity_sum[np.newaxis, :]
    n_sum   = quantity_sum.shape[0]
    q_sum_s = quantity_sum[:, mask][:, sort_idx]

    n_res_sc = state.get('n_resample_sc', 0)
    if n_res_sc > 0:
        # Fast path: pre-scattered positions already in state — just histogram.
        # Identical scatter positions are reused across all calls (consistent PSF).
        from fast_histogram import histogram2d as _h2d
        x_sc    = state['x_sc']
        y_sc    = state['y_sc']
        out_sum = np.empty((n_sum, n_pix, n_pix), dtype=np.float32)
        for i in range(n_sum):
            w = np.repeat(q_sum_s[i], n_res_sc) / n_res_sc
            out_sum[i] = _h2d(x_sc, y_sc, weights=w, bins=n_pix, range=range2d)
    else:
        out_sum, _, _ = _run_scatter(
            x_s, y_s, dist_s, q_sum_s, None,
            extent, n_pix, n_sum, 0, n_resample, scatter_batch, njobs, verbose=False,
        )

    x_hi         = state['x_hi']
    y_hi         = state['y_hi']
    n_unsmoothed = len(x_hi)
    sigma_aa     = 1.0 if antialias else 0.0

    imgs = []
    for i in range(n_sum):
        h = out_sum[i].copy()
        if n_unsmoothed > 0:
            h += histogram2d(x_hi, y_hi,
                             weights=quantity_sum[i][~mask],
                             bins=n_pix, range=range2d)
        h[~np.isfinite(h)] = 0.0
        imgs.append(_gaussian_smooth_2d(h, sigma_aa) if sigma_aa > 0 else h)

    return imgs


# Public API: 3-D density cube

def build_density_cube(X, mass, res=1.0, extent=None, k=5, nsteps=None,
                       njobs=1, sigma_step=0.5, sigma_min=0.5, verbose=True,
                       method='scatter', n_resample=100, scatter_batch=1000):
    """Adaptively smooth particles onto a 3-D density cube.

    Uses a scatter-only kNN smoothing algorithm in three dimensions.

    Parameters
    ----------
    X : array, shape (3, N)
        3-D particle positions in any consistent length unit (e.g. kpc).
    mass : array, shape (N,)
        Scalar quantity to accumulate per voxel (e.g. mass in M_sun).
    res : float
        Voxel size [same units as X].
    extent : float, optional
        Half-size of the cube; grid spans ``[-extent, +extent]`` per axis.
        Defaults to the maximum absolute coordinate value.
    k : int
        Number of nearest neighbours for smoothing-length estimation.
    nsteps : int, optional
        Deprecated and ignored.
    njobs : int
        Parallel threads for the kNN query (scipy backend only).
    sigma_step : float, optional
        Deprecated and ignored.
    sigma_min : float, optional
        Deprecated and ignored.

    Returns
    -------
    cube : ndarray, shape (n_pix, n_pix, n_pix), float32
        Quantity density in units of [mass units] / [length units]^3.
    """
    import warnings
    if nsteps is not None:
        warnings.warn(
            "nsteps is deprecated and ignored.",
            DeprecationWarning, stacklevel=2,
        )

    if sigma_step != 0.5 or sigma_min != 0.5:
        warnings.warn(
            "sigma_step and sigma_min are deprecated and ignored in scatter-only mode.",
            DeprecationWarning, stacklevel=2,
        )

    if method != 'scatter':
        raise ValueError("Only method='scatter' is supported.")

    X    = np.asarray(X,    dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).ravel()

    if extent is None:
        extent = float(np.max(np.abs(X)))
    n_pix = int(extent * 2.0 / res) - 1
    # voxel edges and spacing (uniform grid)
    lo   = -extent
    step = (2.0 * extent) / n_pix

    _, dist = nearest_neighbour_density(X.T, k=k, njobs=njobs)

    sort_idx = np.argsort(dist)
    xs       = X[0][sort_idx]
    ys       = X[1][sort_idx]
    zs       = X[2][sort_idx]
    dist_s   = dist[sort_idx]
    mass_s   = mass[sort_idx]

    N = len(dist_s)

    if verbose:
        print(f"[INFO] Density cube scatter: {N} particles, {n_resample} resamples, "
              f"batch={scatter_batch} ({N * n_resample:,} total sub-particles).")
    rng  = np.random.default_rng()
    out  = np.zeros((n_pix, n_pix, n_pix), dtype=np.float64)
    _iter = range(0, N, scatter_batch)
    if verbose:
        try:
            import tqdm
            _iter = tqdm.tqdm(_iter, total=(N - 1) // scatter_batch + 1)
        except ImportError:
            pass
    for b0 in _iter:
        b1   = min(b0 + scatter_batch, N)
        bm   = b1 - b0
        tot  = bm * n_resample
        sigs = np.repeat(dist_s[b0:b1].astype(np.float32), n_resample)
        dx   = rng.standard_normal(tot).astype(np.float32) * sigs
        dy   = rng.standard_normal(tot).astype(np.float32) * sigs
        dz_  = rng.standard_normal(tot).astype(np.float32) * sigs
        xb   = np.repeat(xs[b0:b1].astype(np.float32), n_resample) + dx
        yb   = np.repeat(ys[b0:b1].astype(np.float32), n_resample) + dy
        zb   = np.repeat(zs[b0:b1].astype(np.float32), n_resample) + dz_
        wb   = np.repeat(mass_s[b0:b1].astype(np.float32), n_resample) / n_resample
        ix   = ((xb - lo) / step).astype(np.intp)
        iy   = ((yb - lo) / step).astype(np.intp)
        iz   = ((zb - lo) / step).astype(np.intp)
        ok   = ((ix >= 0) & (ix < n_pix) &
                (iy >= 0) & (iy < n_pix) &
                (iz >= 0) & (iz < n_pix))
        flat = ix[ok] * n_pix * n_pix + iy[ok] * n_pix + iz[ok]
        out += np.bincount(flat, weights=wb[ok].astype(np.float64),
                           minlength=n_pix ** 3).reshape(n_pix, n_pix, n_pix)
    out /= res ** 3
    return out.astype(np.float32)
