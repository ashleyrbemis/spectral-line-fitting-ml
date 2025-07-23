import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import pandas as pd
import json

def gaussian(x, amplitude, mean, sigma):
    if np.isclose(amplitude, 0.0, atol=1e-9): # Check if amplitude is effectively zero
        return np.zeros_like(x)
    if np.isclose(sigma, 0.0, atol=1e-9):
        sigma = 1e-9
    return amplitude * np.exp(-((x - mean)**2) / (2 * sigma**2))

def generate_synthetic_spectrum(
    x_range: tuple = (0.0, 100.0),
    num_channels: int = 256,
    components: list = None,
    rms_noise_level: float = 0.1,
    seed: int = None
) -> dict:
    """
    Generates a synthetic spectrum with multiple Gaussian components and added Gaussian noise.

    Parameters
    ----------
    x_range : tuple, optional
        (min_x, max_x) for the spectral axis. Defaults to (0.0, 100.0).
    num_channels : int, optional
        Number of data points (channels) in the spectrum. Defaults to 256.
    components : list, optional
        A list of dictionaries, where each dictionary defines a Gaussian component:
        {'amplitude': float, 'mean': float, 'sigma': float}.
        If None, a default set of components will be used.
    rms_noise_level : float, optional
        The RMS (root-mean-square) noise level to add to the spectrum. Defaults to 0.1.
    seed : int, optional
        Random seed for reproducibility of noise. If None, results will vary.

    Returns
    -------
    dict
        A dictionary containing:
        'x': np.ndarray (spectral axis)
        'y_true': np.ndarray (true, noiseless spectrum)
        'y_observed': np.ndarray (spectrum with noise)
        'rms_noise': float (the RMS noise level used)
        'chan_wid': float (channel width)
        'components_info': list (the list of components used to generate the spectrum)
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(x_range[0], x_range[1], num_channels)
    chan_wid = (x_range[1] - x_range[0]) / (num_channels - 1)

    if components is None:
        components = [{'amplitude': 5.0, 'mean': 40.0, 'sigma': 3.0}]
        print("Using default single Gaussian component.")

    y_true = np.zeros_like(x)
    for comp in components:
        amp = comp['amplitude']
        mean = comp['mean']
        sigma = comp['sigma']

        if sigma <= 0:
            print(f"Warning: Component with non-positive sigma ({sigma}) skipped.")
            continue

        y_true += amp * np.exp(-((x - mean)**2) / (2 * sigma**2))

    noise = norm.rvs(loc=0, scale=rms_noise_level, size=num_channels)
    y_observed = y_true + noise

    return {
        'x': x,
        'y_true': y_true,
        'y_observed': y_observed,
        'rms_noise': rms_noise_level,
        'chan_wid': chan_wid,
        'components_info': components
    }

def generate_and_save_test_spectra(output_dir='synthetic_spectra_test_data'):
    """
    Generates, plots with true peaks, and saves a suite of synthetic spectra for various test cases,
    including detailed peak information to CSV.

    Parameters
    ----------
    output_dir : str, optional
        The directory where synthetic spectra data and plots will be saved.
        Defaults to 'synthetic_spectra_test_data'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory '{output_dir}' already exists. Overwriting contents (CSV/JSON/PNGs).")

    # Define all test cases with their parameters based on your detailed request
    test_cases_config = {
        "1_single_strong_peak": {
            "components": [{'amplitude': 10.0, 'mean': 50.0, 'sigma': 2.0}],
            "rms_noise_level": 0.1, "seed": 201,
            "description": "1. Single, strong peak."
        },
        "2_multiple_strong_separated_peaks": {
            "components": [
                {'amplitude': 8.0, 'mean': 30.0, 'sigma': 2.0},
                {'amplitude': 9.0, 'mean': 70.0, 'sigma': 3.0}
            ],
            "rms_noise_level": 0.15, "seed": 202,
            "description": "2. Multiple strong, separated peaks."
        },
        "3a_strong_slightly_blended_3comp": {
            "components": [
                {'amplitude': 7.0, 'mean': 45.0, 'sigma': 2.5},
                {'amplitude': 6.0, 'mean': 47.0, 'sigma': 3.0},
                {'amplitude': 5.0, 'mean': 49.0, 'sigma': 2.0}
            ],
            "rms_noise_level": 0.2, "seed": 203,
            "description": "3. Multiple strong, slightly blended peaks (3 components)."
        },
        "3b_strong_slightly_blended_5comp": {
            "components": [
                {'amplitude': 7.0, 'mean': 35.0, 'sigma': 2.0},
                {'amplitude': 6.0, 'mean': 37.0, 'sigma': 2.5},
                {'amplitude': 5.0, 'mean': 40.0, 'sigma': 1.8},
                {'amplitude': 4.0, 'mean': 43.0, 'sigma': 2.2},
                {'amplitude': 3.0, 'mean': 45.0, 'sigma': 1.5}
            ],
            "rms_noise_level": 0.25, "seed": 204,
            "description": "3. Multiple strong, slightly blended peaks (5 components)."
        },
        "4a_strong_severely_blended_3comp": {
            "components": [
                {'amplitude': 8.0, 'mean': 50.0, 'sigma': 3.0},
                {'amplitude': 7.0, 'mean': 51.0, 'sigma': 3.5},
                {'amplitude': 6.0, 'mean': 52.0, 'sigma': 4.0}
            ],
            "rms_noise_level": 0.3, "seed": 205,
            "description": "4. Multiple strong, severely blended peaks (3 components)."
        },
        "4b_strong_severely_blended_5comp": {
            "components": [
                {'amplitude': 7.0, 'mean': 40.0, 'sigma': 3.0},
                {'amplitude': 6.0, 'mean': 41.0, 'sigma': 2.5},
                {'amplitude': 5.0, 'mean': 42.0, 'sigma': 2.0},
                {'amplitude': 4.0, 'mean': 43.0, 'sigma': 1.5},
                {'amplitude': 3.0, 'mean': 44.0, 'sigma': 1.0}
            ],
            "rms_noise_level": 0.4, "seed": 206,
            "description": "4. Multiple strong, severely blended peaks (5 components)."
        },
        "5_low_snr_single_peak": {
            "components": [{'amplitude': 1.0, 'mean': 50.0, 'sigma': 2.0}],
            "rms_noise_level": 0.5, # High noise, low amplitude
            "seed": 207,
            "description": "5. Low signal-to-noise single peak."
        },
        "6_low_snr_multiple_separated_peaks": {
            "components": [
                {'amplitude': 1.2, 'mean': 30.0, 'sigma': 2.0},
                {'amplitude': 1.0, 'mean': 70.0, 'sigma': 3.0}
            ],
            "rms_noise_level": 0.5, "seed": 208,
            "description": "6. Multiple low signal-to-noise well-separated peaks."
        },
        "7a_low_snr_slightly_blended_3comp": {
            "components": [
                {'amplitude': 1.0, 'mean': 45.0, 'sigma': 2.5},
                {'amplitude': 0.8, 'mean': 47.0, 'sigma': 3.0},
                {'amplitude': 0.7, 'mean': 49.0, 'sigma': 2.0}
            ],
            "rms_noise_level": 0.5, "seed": 209,
            "description": "7. Multiple low signal-to-noise slightly blended peaks (3 components)."
        },
        "7b_low_snr_slightly_blended_5comp": {
            "components": [
                {'amplitude': 1.0, 'mean': 35.0, 'sigma': 2.0},
                {'amplitude': 0.9, 'mean': 37.0, 'sigma': 2.5},
                {'amplitude': 0.8, 'mean': 40.0, 'sigma': 1.8},
                {'amplitude': 0.7, 'mean': 43.0, 'sigma': 2.2},
                {'amplitude': 0.6, 'mean': 45.0, 'sigma': 1.5}
            ],
            "rms_noise_level": 0.55, "seed": 210,
            "description": "7. Multiple low signal-to-noise slightly blended peaks (5 components)."
        },
        "8a_low_snr_severely_blended_3comp": {
            "components": [
                {'amplitude': 1.0, 'mean': 50.0, 'sigma': 3.0},
                {'amplitude': 0.9, 'mean': 51.0, 'sigma': 3.5},
                {'amplitude': 0.8, 'mean': 52.0, 'sigma': 4.0}
            ],
            "rms_noise_level": 0.6, "seed": 211,
            "description": "8. Multiple low signal-to-noise severely blended peaks (3 components)."
        },
        "8b_low_snr_severely_blended_5comp": {
            "components": [
                {'amplitude': 1.0, 'mean': 40.0, 'sigma': 3.0},
                {'amplitude': 0.9, 'mean': 41.0, 'sigma': 2.5},
                {'amplitude': 0.8, 'mean': 42.0, 'sigma': 2.0},
                {'amplitude': 0.7, 'mean': 43.0, 'sigma': 1.5},
                {'amplitude': 0.6, 'mean': 44.0, 'sigma': 1.0}
            ],
            "rms_noise_level": 0.65, "seed": 212,
            "description": "8. Multiple low signal-to-noise severely blended peaks (5 components)."
        },
        "pure_noise_spectrum": {
            "components": [], # No components, just noise
            "rms_noise_level": 0.5, "seed": 213,
            "description": "Spectrum with only noise, no true signal."
        },
        "edge_peak_spectrum": {
            "components": [
                {'amplitude': 8.0, 'mean': 5.0, 'sigma': 2.0}, # Peak near the start of the spectrum
                {'amplitude': 7.0, 'mean': 95.0, 'sigma': 2.5}  # Peak near the end of the spectrum
            ],
            "rms_noise_level": 0.2, "seed": 214,
            "description": "Peaks near the edges of the spectral range."
        }
    }

    test_case_info_list = [] # To store metadata for a summary CSV

    for name, params in test_cases_config.items():
        print(f"\n--- Generating '{name}' ---")

        # Get parameters, setting defaults if not specified for a test case
        x_range = params.get("x_range", (0.0, 100.0))
        num_channels = params.get("num_channels", 256)

        synth_data = generate_synthetic_spectrum(
            x_range=x_range,
            num_channels=num_channels,
            components=params['components'],
            rms_noise_level=params['rms_noise_level'],
            seed=params['seed']
        )

        # Calculate FWHM for each component
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.3548 * sigma
        FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))
        
        # Prepare data for the true_peaks_info CSV
        true_peaks_data = []
        if synth_data['components_info']: # Only if there are actual components
            for i, comp in enumerate(synth_data['components_info']):
                true_peaks_data.append({
                    'component_idx': i + 1,
                    'true_amplitude': comp['amplitude'],
                    'true_mean': comp['mean'],
                    'true_sigma': comp['sigma'],
                    'true_fwhm': comp['sigma'] * FWHM_FACTOR
                })
        
        df_true_peaks = pd.DataFrame(true_peaks_data)
        
        # Add overall spectrum info to the true_peaks_df (repeated for each row or as metadata)
        if not df_true_peaks.empty:
            df_true_peaks['num_true_peaks'] = len(synth_data['components_info'])
            df_true_peaks['spectrum_rms_noise'] = synth_data['rms_noise']
        else: # Handle case with no components (pure noise)
            df_true_peaks = pd.DataFrame([{
                'component_idx': np.nan, 'true_amplitude': np.nan,
                'true_mean': np.nan, 'true_sigma': np.nan, 'true_fwhm': np.nan,
                'num_true_peaks': 0, 'spectrum_rms_noise': synth_data['rms_noise']
            }])


        # 1. Save spectrum data (x, y_observed, y_true) to a CSV
        df_spectrum = pd.DataFrame({
            'x': synth_data['x'],
            'y_observed': synth_data['y_observed'],
            'y_true': synth_data['y_true']
        })
        data_filepath = os.path.join(output_dir, f'{name}_spectrum_data.csv')
        df_spectrum.to_csv(data_filepath, index=False)
        print(f"  Saved spectrum data to: {data_filepath}")

        # 2. Save true components info (ground truth) to a JSON (for raw parameters)
        components_filepath = os.path.join(output_dir, f'{name}_true_components.json')
        with open(components_filepath, 'w') as f:
            json.dump(synth_data['components_info'], f, indent=4)
        print(f"  Saved true components info to: {components_filepath}")

        # 3. Save detailed true peak information to a CSV
        true_peaks_csv_filepath = os.path.join(output_dir, f'{name}_true_peaks_info.csv')
        df_true_peaks.to_csv(true_peaks_csv_filepath, index=False)
        print(f"  Saved true peaks summary to: {true_peaks_csv_filepath}")

        # 4. Generate and save a plot with true peak indicators
        plt.figure(figsize=(12, 6))
        plt.plot(synth_data['x'], synth_data['y_true'], label='True Noiseless Spectrum', linestyle='--', color='blue', linewidth=2)
        plt.plot(synth_data['x'], synth_data['y_observed'], label='Observed Spectrum (with noise)', color='red', alpha=0.7, linewidth=1)
        plt.axhline(synth_data['rms_noise'], color='gray', linestyle=':', label=f'Input RMS Noise ({synth_data["rms_noise"]:.2f})')
        plt.axhline(-synth_data['rms_noise'], color='gray', linestyle=':')

        # Plot individual true Gaussian components
        if synth_data['components_info']:
            for i, comp in enumerate(synth_data['components_info']):
                gaussian_component = comp['amplitude'] * np.exp(-((synth_data['x'] - comp['mean'])**2) / (2 * comp['sigma']**2))
                plt.plot(synth_data['x'], gaussian_component, color='green', linestyle='-', alpha=0.7, label=f'True Comp {i+1}' if i==0 else "", linewidth=1.5)
                plt.plot(comp['mean'], comp['amplitude'], 'x', color='black', markersize=8, markeredgewidth=2) # Mark the peak center
                # You could add vertical lines for FWHM here if desired:
                # plt.axvline(comp['mean'] - comp['sigma']*np.sqrt(2*np.log(2)), color='purple', linestyle=':', alpha=0.5)
                # plt.axvline(comp['mean'] + comp['sigma']*np.sqrt(2*np.log(2)), color='purple', linestyle=':', alpha=0.5)


        plt.title(f'Synthetic Spectrum: {name.replace("_", " ").title()} ({len(params["components"])} Comp(s))')
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Intensity (Jy/beam)')
        plt.legend()
        plt.grid(True)
        plot_filepath = os.path.join(output_dir, f'{name}_spectrum_plot.png')
        plt.savefig(plot_filepath, bbox_inches='tight')
        plt.close() # Close the plot to free memory
        print(f"  Saved spectrum plot with true peaks to: {plot_filepath}")

        # Add info to a list for overall summary
        test_case_info_list.append({
            'test_case_name': name,
            'description': params['description'],
            'num_true_components': len(params['components']),
            'input_rms_noise': params['rms_noise_level'],
            'seed': params['seed'],
            'spectrum_data_filepath': data_filepath,
            'true_components_json_filepath': components_filepath,
            'true_peaks_info_csv_filepath': true_peaks_csv_filepath,
            'spectrum_plot_filepath': plot_filepath
        })

    # Save an overall summary CSV
    df_summary = pd.DataFrame(test_case_info_list)
    summary_filepath = os.path.join(output_dir, 'synthetic_spectra_summary.csv')
    df_summary.to_csv(summary_filepath, index=False)
    print(f"\n--- All synthetic spectra generated and saved to '{output_dir}'. ---")
    print(f"Summary of all test cases saved to: {summary_filepath}")

if __name__ == '__main__':
    generate_and_save_test_spectra()