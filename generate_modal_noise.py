"""
Modal noise spectrum simulator.

Generates a synthetic fibre-spectrograph spectrum where the signal is the
sum of a correlated Gaussian Process (modal noise) and white noise.

Physics model
-------------
Modal noise amplitude grows with wavelength because the number of fibre modes
scales as (D/λ)², so fewer modes → more speckle contrast at longer λ.
The GP is drawn with unit amplitude on a log-λ (uniform velocity) grid, then
each pixel is scaled by:

    amp(λ) = amplitude_at_1650nm × (λ / 1650 nm)

White noise is added afterwards.

YAML config keys (all optional — defaults shown)
-------------------------------------------------
wavelength_start_nm : 1000.0   # start of wavelength range [nm]
wavelength_end_nm   : 1800.0   # end of wavelength range   [nm]
velocity_step_kms   : 1.0      # pixel spacing              [km/s]
kernel_width_kms    : 3.5      # GP correlation length      [km/s]
amplitude_at_1650nm : 0.006    # GP RMS amplitude at 1650 nm
white_noise         : 0.005    # white-noise std per pixel

Standalone usage
----------------
python generate_modal_noise.py [config.yaml] [--outfile out.csv] [--seed N] [--plot]

Importable usage
----------------
from generate_modal_noise import generate_spectrum

wave, spectrum, gp_signal = generate_spectrum()
wave, spectrum, gp_signal = generate_spectrum(yaml_file='my_config.yaml', seed=42)
wave, spectrum, gp_signal = generate_spectrum(wave_grid=my_wave_nm,
                                               params={'kernel_width_kms': 10})
"""

import argparse
import numpy as np
import yaml

C_LIGHT = 299792.458  # km/s

DEFAULTS = {
    'wavelength_start_nm': 1000.0,
    'wavelength_end_nm':   1800.0,
    'velocity_step_kms':   1.0,
    'kernel_width_kms':    3.5,
    'amplitude_at_1650nm': 0.006,
    'white_noise':         0.005,
}


# ── internal draw helpers ────────────────────────────────────────────────────

def _draw_gp_fft(N, dv, length, rng):
    """
    Draw unit-amplitude GP on a UNIFORM velocity grid.
    Uses FFT circulant embedding — O(N log N), no Cholesky, numerically safe.
    """
    m    = int(2 ** np.ceil(np.log2(2 * N)))   # next power of 2 ≥ 2N
    idx  = np.arange(m)
    dist = np.minimum(idx, m - idx)             # circulant distance in pixels
    c    = np.exp(-0.5 * (dist * dv) ** 2 / length ** 2)
    lam  = np.maximum(np.real(np.fft.rfft(c)), 0.0)  # eigenvalues, clip ≥ 0
    z    = rng.standard_normal(m)
    return np.fft.irfft(np.sqrt(lam) * np.fft.rfft(z), n=m)[:N]


def _draw_gp_banded(vel, pixel_scale, length, rng):
    """
    Draw unit-amplitude GP on a NON-UNIFORM velocity grid.
    Uses banded Cholesky — suitable for short custom grids.
    """
    from scipy.linalg import cholesky_banded
    N  = len(vel)
    bw = min(N - 1, int(np.ceil(6.0 * length / pixel_scale)))
    ab = np.zeros((bw + 1, N))
    for k in range(bw + 1):
        dv = vel[k:] - vel[:N - k]
        ab[k, :N - k] = np.exp(-0.5 * dv ** 2 / length ** 2)
    ab[0, :] += 1e-6   # jitter for numerical positive definiteness
    cb   = cholesky_banded(ab, lower=True)
    z    = rng.standard_normal(N)
    draw = np.zeros(N)
    for k in range(bw + 1):
        draw[k:] += cb[k, :N - k] * z[:N - k]
    return draw


# ── public API ───────────────────────────────────────────────────────────────

def generate_spectrum(yaml_file=None, params=None, wave_grid=None, seed=None):
    """
    Generate one realisation of a modal-noise + white-noise spectrum.

    Parameters
    ----------
    yaml_file : str, optional
        Path to YAML config.  Missing keys fall back to DEFAULTS.
    params : dict, optional
        Key/value overrides applied on top of YAML / DEFAULTS.
    wave_grid : array_like, optional
        Custom wavelength grid [nm].  When given, wavelength_start/end/step
        from the config are ignored; banded Cholesky is used instead of FFT.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    wave : ndarray
        Wavelength grid [nm].
    spectrum : ndarray
        Normalised spectrum (GP modal noise + white noise, continuum ≈ 1).
    gp_signal : ndarray
        GP modal-noise component only (no white noise, continuum ≈ 0).
    """
    cfg = dict(DEFAULTS)
    if yaml_file is not None:
        with open(yaml_file) as f:
            cfg.update(yaml.safe_load(f) or {})
    if params is not None:
        cfg.update(params)

    rng = np.random.default_rng(seed)

    # ── wavelength + velocity grids ──────────────────────────────────────────
    if wave_grid is not None:
        wave = np.asarray(wave_grid, dtype=float)
        vel  = C_LIGHT * np.log(wave / wave[0])
    else:
        dv   = float(cfg['velocity_step_kms'])
        lam0 = float(cfg['wavelength_start_nm'])
        lam1 = float(cfg['wavelength_end_nm'])
        N    = int(np.log(lam1 / lam0) * C_LIGHT / dv) + 1
        vel  = np.arange(N, dtype=float) * dv
        wave = lam0 * np.exp(vel / C_LIGHT)

    N           = len(wave)
    pixel_scale = float(np.median(np.diff(vel)))
    length      = float(cfg['kernel_width_kms'])

    # ── draw unit-amplitude GP ───────────────────────────────────────────────
    if wave_grid is None:
        draw = _draw_gp_fft(N, float(cfg['velocity_step_kms']), length, rng)
    else:
        draw = _draw_gp_banded(vel, pixel_scale, length, rng)

    # ── wavelength-dependent amplitude: amp ∝ λ, normalised at 1650 nm ──────
    amp_scale = float(cfg['amplitude_at_1650nm']) * (wave / 1650.0)
    gp_signal = draw * amp_scale

    # ── add white noise ──────────────────────────────────────────────────────
    spectrum = 1.0 + gp_signal + rng.standard_normal(N) * float(cfg['white_noise'])

    return wave, spectrum, gp_signal


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('yaml_file', nargs='?', default=None,
                        help='YAML config file (optional; uses defaults if omitted)')
    parser.add_argument('--outfile', default=None,
                        help='Save to CSV: wavelength_nm, spectrum, gp_signal')
    parser.add_argument('--seed',    type=int, default=None,
                        help='RNG seed for reproducibility')
    parser.add_argument('--plot',    action='store_true',
                        help='Show a quick diagnostic plot')
    args = parser.parse_args()

    wave, spectrum, gp_signal = generate_spectrum(yaml_file=args.yaml_file,
                                                  seed=args.seed)

    if args.outfile:
        header = 'wavelength_nm,spectrum,gp_signal'
        np.savetxt(args.outfile,
                   np.column_stack([wave, spectrum, gp_signal]),
                   delimiter=',', header=header, comments='')
        print(f'Saved {args.outfile}')

    if args.plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(wave, spectrum,      color='0.5', lw=0.4, label='spectrum')
        axes[0].plot(wave, 1 + gp_signal, color='C0', lw=0.9, label='GP signal')
        axes[0].set_ylabel('Normalised flux')
        axes[0].legend(fontsize=9)
        axes[1].plot(wave, gp_signal, color='C0', lw=0.7)
        axes[1].axhline(0, color='k', lw=0.5, ls='--')
        axes[1].set_xlabel('Wavelength [nm]')
        axes[1].set_ylabel('Modal noise amplitude')
        plt.tight_layout()
        plt.show()
