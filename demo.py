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
C_LIGHT    = 299792.458  # km/s

plt.rcParams.update({'font.size': 11, 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in'})

# ── figure 1: single realisation, full range ─────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)

wave, spec, gp = generate_spectrum(seed=7)

ax = axes[0]
ax.plot(wave, spec,    color='0.55', lw=0.35, label='spectrum (GP + noise)')
ax.plot(wave, 1 + gp, color='#2980b9', lw=0.8, label='GP modal noise')
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

# ── figure 2: 3 zoomed regions (5 × kernel_width each), 5 realisations ───────
kernel_width = DEFAULTS['kernel_width_kms']
zoom_kms     = 10 * kernel_width          # total window width in velocity
zoom_centers = [1050.0, 1250.0, 1650.0]  # nm  (Y, J, H)
zoom_labels  = ['Y band (~1050 nm)', 'J band (~1250 nm)', 'H band (~1650 nm)']

# pre-generate 5 realisations
reals = [generate_spectrum(seed=s) for s in range(5)]

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)

for ax, center, label in zip(axes, zoom_centers, zoom_labels):
    half_lam = center * (zoom_kms / 2) / C_LIGHT
    lam_lo, lam_hi = center - half_lam, center + half_lam

    for i, (wave, spec, gp) in enumerate(reals):
        mask = (wave >= lam_lo) & (wave <= lam_hi)
        vel_rel = (wave[mask] / center - 1) * C_LIGHT
        # underlying GP (smooth)
        ax.plot(vel_rel, 1 + gp[mask], color=COLORS[i], lw=2.0, alpha=0.9,
                label=f'#{i+1}' if ax is axes[0] else None)
        # white noise on top
        ax.plot(vel_rel, spec[mask], color=COLORS[i], lw=0.6, alpha=0.45)

    for sign in (-1, 1):
        ax.axvline(sign * kernel_width, color='0.6', lw=0.8, ls='--')
    ax.axvline(0, color='k', lw=0.5, ls=':')

    ax.set_xlim(-zoom_kms / 2, zoom_kms / 2)
    ax.set_ylabel('Normalised flux')
    ax.set_title(label, fontsize=10, loc='left', pad=4)

    if ax is axes[0]:
        ax.legend(fontsize=8, ncol=5, loc='upper right', title='realisation')

# kernel width arrow on middle panel
ax_mid = axes[1]
ybot = ax_mid.get_ylim()[0]
ax_mid.annotate('', xy=(kernel_width, ybot), xytext=(-kernel_width, ybot),
                arrowprops=dict(arrowstyle='<->', color='0.45', lw=1.0),
                annotation_clip=False)
ax_mid.text(0, ybot, f' σ = {kernel_width:.1f} km/s',
            va='top', ha='center', fontsize=8, color='0.45')

axes[-1].set_xlabel('Velocity relative to band centre [km/s]')

plt.tight_layout()
fig.savefig('figures/demo_realizations.png', dpi=150, bbox_inches='tight')
plt.close()
print('figures/demo_realizations.png')

print('\nAll figures written to figures/')
