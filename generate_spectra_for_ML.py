# %%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import random

# Assuming generate_synthetic_spectrum is available from generate_synthetic_spectra.py
# If not, ensure it's defined or imported. For this example, I'll include a minimal definition.
try:
    from generate_synthetic_spectra import generate_synthetic_spectrum, generate_and_save_test_spectra, gaussian
except ImportError:
    print("Warning: 'generate_synthetic_spectra' not found.")

# --- Configuration ---
import config

def generate_ml_training_data(
    num_spectra: int = 10000,
    min_components: int = 1, # REFINEMENT 1: Added min_components
    max_components: int = 5,
    x_range: tuple[float, float] = (0.0, 100.0),
    num_channels: int = 256,
    amplitude_range: tuple[float, float] = (0.5, 15.0),
    mean_range: tuple[float, float] = (10.0, 90.0),
    sigma_range: tuple[float, float] = (1.0, 5.0),
    rms_noise_range: tuple[float, float] = (0.05, 1.0),
    output_filepath: str = 'ml_training_spectra.h5',
    plot_sample_spectra: int = 5,
    global_seed: int | None = None # REFINEMENT 4: For reproducibility
) -> None:
    """
    Generates a large dataset of synthetic spectra for ML training. (Docstring unchanged)
    """
    print(f"Generating {num_spectra} synthetic spectra for ML training...")

    # REFINEMENT 4: Set global seed for reproducible datasets if provided
    if global_seed is not None:
        print(f"Using global random seed: {global_seed}")
        random.seed(global_seed)
        np.random.seed(global_seed)

    all_spectra = []
    all_true_params_ml_format = []
    all_spectrum_metadata = []

    # Create a directory for saving plot samples
    plot_output_dir = config.PLOT_SAVE_DIR
    os.makedirs(plot_output_dir, exist_ok=True)
    print(f"Sample plots will be saved to '{plot_output_dir}/'")

    x_channels = np.linspace(x_range[0], x_range[1], num_channels)
    chan_wid = (x_range[1] - x_range[0]) / (num_channels - 1)
    max_params_per_spectrum = max_components * 3

    # REFINEMENT 3: Use tqdm for a clean progress bar
    for i in tqdm(range(num_spectra), desc="Generating Spectra"):
        
        # REFINEMENT 1: Use min_components
        n_components = random.randint(min_components, max_components)

        components_list_physical = []
        for _ in range(n_components):
            amplitude = np.random.uniform(*amplitude_range)
            mean_physical = np.random.uniform(*mean_range)
            sigma_physical = np.random.uniform(max(sigma_range[0], chan_wid * 0.5), sigma_range[1])
            components_list_physical.append({
                'amplitude': amplitude, 'mean': mean_physical, 'sigma': sigma_physical
            })

        components_list_physical.sort(key=lambda x: x['mean'])

        true_params_ml_format_flat = []
        for comp_phys in components_list_physical:
            mean_index = np.argmin(np.abs(x_channels - comp_phys['mean']))
            sigma_in_channels = comp_phys['sigma'] / chan_wid
            true_params_ml_format_flat.extend([comp_phys['amplitude'], float(mean_index), sigma_in_channels])

        # REFINEMENT 2: Pad with 0.0 instead of np.nan for better framework compatibility
        padding_value = 0.0
        padded_params_ml_format = true_params_ml_format_flat + [padding_value] * (max_params_per_spectrum - len(true_params_ml_format_flat))

        rms_noise = np.random.uniform(*rms_noise_range)
        synth_data = generate_synthetic_spectrum(
            x_range=x_range, num_channels=num_channels, components=components_list_physical,
            rms_noise_level=rms_noise, seed=None
        )

        all_spectra.append(synth_data['y_observed'])
        all_true_params_ml_format.append(padded_params_ml_format)
        all_spectrum_metadata.append({
            'num_true_components': n_components, 'rms_noise': rms_noise, 'chan_wid': chan_wid,
            'x_min': x_range[0], 'x_max': x_range[1],
            'raw_components_json': json.dumps(components_list_physical)
        })
        
        # Save sample plots to a file instead of displaying them
        if i < plot_sample_spectra:
            plt.figure(figsize=(10, 5))
            plt.plot(synth_data['x'], synth_data['y_true'], label='True Noiseless Spectrum')
            plt.plot(synth_data['x'], synth_data['y_observed'], label='Observed (with noise)', alpha=0.7)
            plt.title(f'Sample Spectrum {i+1} ({n_components} True Components)')
            plt.xlabel('Channel')
            plt.ylabel('Intensity')
            plt.legend()
            plt.grid(True)
            
            # Define the output path and save the figure
            plot_filename = os.path.join(plot_output_dir, f'sample_spectrum_{i+1}.png')
            plt.savefig(plot_filename)
            plt.close() # Close the figure to free up memory and prevent it from popping up

    # --- Saving data (logic is unchanged) ---
    store = pd.HDFStore(output_filepath, mode='w')
    store['spectra'] = pd.DataFrame(np.array(all_spectra))
    store['true_params_ml_format'] = pd.DataFrame(all_true_params_ml_format, columns=[f'param_{j}' for j in range(max_params_per_spectrum)])
    store['metadata'] = pd.DataFrame(all_spectrum_metadata)
    store['x_channels_template'] = pd.DataFrame({'x': x_channels})
    store.close()

    print(f"\nGenerated {num_spectra} spectra and saved to {output_filepath}")

# --- Main execution block ---
if __name__ == '__main__':
    import sys
    import os
    # Ensure current script's directory is in sys.path to find config.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    import config # Import your configuration

    print("--- Starting Data Generation Tasks ---")

    print("\n--- Generating Full Training Data ---")
    generate_ml_training_data(
        num_spectra = config.NUM_TRAINING_SPECTRA, 
        min_components = config.MIN_COMPONENTS_PER_SPECTRUM,
        max_components = config.MAX_COMPONENTS_IN_DATA,  
        amplitude_range=(config.COMP_AMP_MIN, config.COMP_AMP_MAX),
        num_channels = config.NUM_CHANNELS,
        x_range = (config.DATA_X_MIN, config.DATA_X_MAX),
        mean_range=(config.COMP_MEAN_MIN, config.COMP_MEAN_MAX),
        sigma_range=(config.COMP_SIGMA_MIN, config.COMP_SIGMA_MAX),
        rms_noise_range=(config.RMS_NOISE_MIN, config.RMS_NOISE_MAX),
        output_filepath=config.TRAINING_DATA_FILE, 
        plot_sample_spectra=config.NUM_PLOT_SAMPLE_SPECTRA,
    )
    print("\n--- Full Training Data Generation Complete ---")

    print("\n--- Generating General Validation Data ---")
    generate_ml_training_data(
        num_spectra = config.NUM_VALIDATION_SPECTRA, 
        min_components = config.MIN_COMPONENTS_PER_SPECTRUM,
        max_components = config.MAX_COMPONENTS_IN_DATA,  
        amplitude_range=(config.COMP_AMP_MIN, config.COMP_AMP_MAX),
        num_channels = config.NUM_CHANNELS,
        x_range = (config.DATA_X_MIN, config.DATA_X_MAX),
        mean_range=(config.COMP_MEAN_MIN, config.COMP_MEAN_MAX),
        sigma_range=(config.COMP_SIGMA_MIN, config.COMP_SIGMA_MAX),
        rms_noise_range=(config.RMS_NOISE_MIN, config.RMS_NOISE_MAX),
        output_filepath=config.VALIDATION_DATA_FILE, 
        plot_sample_spectra=config.NUM_PLOT_SAMPLE_SPECTRA, 
    )
    print("\n--- General Validation Data Generation Complete ---")

    print("\n--- All Data Generation Tasks Finished ---")