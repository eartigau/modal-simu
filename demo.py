"""
Generate demo figures for the modal-simu website.
Run once to populate the figures/ folder before deploying.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from generate_modal_noise import generate_spectrum, DEFAULTS

Path('figures').mkdir(exist_ok=True)

WAVE_BANDS = {'Y': (960, 1100), 'J': (1150, 1350), 'H': (1450, 1800)}
COLORS     = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad', '#c0392b']

plt.rcParams.update({'font.size': 11, 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in'})

# ── figure 1: single realisation, full range ─────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)

wave, spec, gp = generate_spectrum(seed=7)

ax = axes[0]
ax.plot(wave, spec,      color='0.55', lw=0.35, label='spectrum (GP + noise)')
ax.plot(wave, 1 + gp,   color='#2980b9', lw=0.8, label='GP modal noise')
ax.set_ylabel('Normalised flux')
ax.legend(fontsize=9, loc='upper left')
ax.set_ylim(0.96, 1.04)
for name, (w0, w1) in WAVE_BANDS.items():
    ax.axvspan(w0, w1, alpha=0.07, color='steelblue')
    ax.text((w0 + w1) / 2, 1.038, name, ha='center', fontsize=8, color='steelblue')

ax = axes[1]
ax.plot(wave, gp, color='#2980b9', lw=0.7)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('Modal noise')
for name, (w0, w1) in WAVE_BANDS.items():
    ax.axvspan(w0, w1, alpha=0.07, color='steelblue')

plt.tight_layout()
fig.savefig('figures/demo_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print('figures/demo_spectrum.png')

# ── figure 2: five realisations overlaid ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))

for i, seed in enumerate(range(5)):
    wave, spec, gp = generate_spectrum(seed=seed)
    ax.plot(wave, spec, color=COLORS[i], lw=0.5, alpha=0.85,
            label=f'realisation {i+1}')

ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('Normalised flux')
ax.legend(fontsize=8, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.12))
ax.set_ylim(0.96, 1.04)
plt.tight_layout()
fig.savefig('figures/demo_realizations.png', dpi=150, bbox_inches='tight')
plt.close()
print('figures/demo_realizations.png')

# ── figure 3: wavelength-dependent amplitude envelope ────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

n_real = 50
wave_ref, _, _ = generate_spectrum(seed=0)
stds = np.zeros((n_real, len(wave_ref)))
for i in range(n_real):
    _, _, gp = generate_spectrum(seed=i)
    stds[i] = gp

rms_env = np.std(stds, axis=0)
expected = DEFAULTS['amplitude_at_1650nm'] * (wave_ref / 1650.0)

ax.plot(wave_ref, rms_env, color='#2980b9', lw=1.2, label='empirical RMS (50 draws)')
ax.plot(wave_ref, expected, color='#e74c3c', lw=1.2, ls='--',
        label=r'model: $A_{1650}\,\times\,\lambda/1650$')
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('Modal noise RMS')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig('figures/demo_amplitude.png', dpi=150, bbox_inches='tight')
plt.close()
print('figures/demo_amplitude.png')

print('\nAll figures written to figures/')
