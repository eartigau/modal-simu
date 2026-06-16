# modal-simu — Modal Noise Simulator

Generates synthetic modal-noise spectra for fibre-fed astronomical spectrographs
using a Gaussian Process model with wavelength-dependent amplitude.

**Website / documentation:** https://eartigau.github.io/modal-simu/

## Installation

```bash
pip install numpy scipy matplotlib pyyaml
```

## Quick start

```python
from generate_modal_noise import generate_spectrum

wave, spectrum, gp_signal = generate_spectrum()                        # defaults
wave, spectrum, gp_signal = generate_spectrum(yaml_file='default_config.yaml', seed=42)
wave, spectrum, gp_signal = generate_spectrum(params={'kernel_width_kms': 10})
```

Or from the command line:

```bash
python generate_modal_noise.py --plot
python generate_modal_noise.py my_config.yaml --outfile spectrum.csv --seed 42
```

## Parameters (`default_config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `wavelength_start_nm` | 1000 | Start of wavelength range [nm] |
| `wavelength_end_nm` | 1800 | End of wavelength range [nm] |
| `velocity_step_kms` | 1.0 | Pixel spacing in velocity [km/s] |
| `kernel_width_kms` | 3.5 | GP correlation length [km/s] |
| `amplitude_at_1650nm` | 0.006 | GP RMS amplitude at 1650 nm |
| `white_noise` | 0.005 | White-noise std per pixel |

## Demo figures

```bash
python demo.py   # writes figures/ used by the website
```

## Author

[Étienne Artigau](https://eartigau.github.io/) — iREx, Université de Montréal
