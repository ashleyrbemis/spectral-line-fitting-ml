# hcop10_utils.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import joblib
import os
import random

# Assume config.py is in a directory accessible via sys.path or the working directory
import config 

# Assume gaussian is in generate_synthetic_spectra.py and accessible
from generate_synthetic_spectra import gaussian 

# --- Helper function for Gaussian (TensorFlow-compatible for use in loss) ---
@tf.function 
def gaussian_tf(x, amplitude, mean, sigma): 
    x = tf.cast(x, tf.float32)
    amplitude = tf.cast(amplitude, tf.float32)
    mean = tf.cast(mean, tf.float32)
    sigma = tf.cast(sigma, tf.float32)

    sigma = tf.where(tf.math.less_equal(sigma, 1e-9), tf.constant(1e-9, dtype=tf.float32), sigma)
    amplitude = tf.where(tf.math.less_equal(amplitude, 1e-9), tf.constant(0.0, dtype=tf.float32), amplitude)

    return amplitude * tf.math.exp(-tf.math.square(x - mean) / (2.0 * tf.math.square(sigma)))


# --- Multi-component Gaussian function (TensorFlow-compatible) ---
@tf.function 
def gaussian_n_components_tf(x, component_params_batch): 
    x = tf.cast(x, tf.float32) 
    component_params_batch = tf.cast(component_params_batch, tf.float32) 

    batch_size = tf.shape(component_params_batch)[0]
    num_channels = tf.shape(x)[0]

    amplitudes = component_params_batch[..., 0] 
    means = component_params_batch[..., 1]      
    sigmas = component_params_batch[..., 2]     

    x_reshaped = tf.reshape(x, (1, 1, num_channels)) 
    amplitudes_reshaped = tf.expand_dims(amplitudes, axis=-1) 
    means_reshaped = tf.expand_dims(means, axis=-1)           
    sigmas_reshaped = tf.expand_dims(sigmas, axis=-1)         

    sigmas_reshaped_safe = tf.where(tf.math.less_equal(sigmas_reshaped, 1e-9), tf.constant(1e-9, dtype=tf.float32), sigmas_reshaped)
    amplitudes_reshaped_safe = tf.where(tf.math.less_equal(amplitudes_reshaped, 1e-9), tf.constant(0.0, dtype=tf.float32), amplitudes_reshaped)

    individual_gaussians_contributions = amplitudes_reshaped_safe * tf.math.exp(
        -tf.math.square(x_reshaped - means_reshaped) / (2.0 * tf.math.square(sigmas_reshaped_safe))
    )

    y_sum_batch = tf.reduce_sum(individual_gaussians_contributions, axis=1) 

    return y_sum_batch

# --- UPDATED: Custom Loss Function with Reconstruction Loss ---
@keras.saving.register_keras_serializable(package="CustomLosses")
def custom_gaussian_loss(x_channels, sparsity_weight=0.01, reconstruction_weight=0.1): 
    """
    Custom loss function for multi-component Gaussian fitting including reconstruction loss.
    Combines MSE for active component parameters with an L1 sparsity penalty for inactive amplitudes
    and an MSE penalty on the reconstructed spectrum.

    Parameters
    ----------
    x_channels : np.ndarray
        The spectral x-axis values (physical units). Used for reconstructing spectra within the loss.
    sparsity_weight : float
        Weight for the L1 penalty on predicted amplitudes of inactive components.
        Higher values encourage more zeros.
    reconstruction_weight : float
        Weight for the MSE penalty between the true and predicted reconstructed spectra.

    Returns
    -------
    callable
        A Keras-compatible loss function.
    """
    # x_channels must be a TF tensor for use in graph. Convert once here in the outer function.
    x_channels_tf = tf.constant(x_channels, dtype=tf.float32)

    def loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        batch_size = tf.shape(y_true)[0]
        
        # --- Parameter-wise Loss ---
        total_param_loss = 0.0 
        # Loop through each potential component slot (e.g., 0 to MAX_COMPONENTS_IN_DATA-1)
        for i_comp in range(config.MAX_COMPONENTS_IN_DATA):
            # Extract true and predicted parameters for the current component slot
            true_amp = y_true[:, i_comp*3]
            true_mean = y_true[:, i_comp*3 + 1]
            true_sigma = y_true[:, i_comp*3 + 2]

            pred_amp = y_pred[:, i_comp*3]
            pred_mean = y_pred[:, i_comp*3 + 1]
            pred_sigma = y_pred[:, i_comp*3 + 2]

            # Create masks: 'active_mask' if component is truly present (true_amp > 1e-6),
            # 'inactive_mask' if it's noise or padded (true_amp <= 1e-6).
            active_mask = tf.cast(tf.greater(true_amp, 1e-6), tf.float32) 
            inactive_mask = 1.0 - active_mask 

            # --- Loss for Active Components (Standard MSE - no individual weights now) ---
            # Calculates MSE for amplitude, mean, sigma, applied only where the component is truly active.
            mse_amp = tf.square(true_amp - pred_amp) * active_mask 
            mse_mean = tf.square(true_mean - pred_mean) * active_mask 
            mse_sigma = tf.square(true_sigma - pred_sigma) * active_mask 
            
            # Sum the MSE terms for active components for this slot for the batch
            total_param_loss += tf.reduce_sum(mse_amp + mse_mean + mse_sigma)

            # --- Loss for Inactive Components (Sparsity Penalty on Amplitude) ---
            # Apply L1 penalty to predicted amplitude only when true component is inactive.
            l1_penalty_on_inactive_amp = tf.abs(pred_amp) * inactive_mask 
            total_param_loss += tf.reduce_sum(l1_penalty_on_inactive_amp * sparsity_weight)

        # --- Reconstruction Loss ---
        # Reshape y_true and y_pred to (batch_size, num_components, 3) for reconstruction
        y_true_reshaped_for_reconstruction = tf.reshape(y_true, (batch_size, config.MAX_COMPONENTS_IN_DATA, 3))
        y_pred_reshaped_for_reconstruction = tf.reshape(y_pred, (batch_size, config.MAX_COMPONENTS_IN_DATA, 3))

        # Reconstruct full spectra for the batch using the TF-compatible functions
        y_true_reconstructed_spectrum_batch = gaussian_n_components_tf(
            x_channels_tf, y_true_reshaped_for_reconstruction 
        )
        y_pred_reconstructed_spectrum_batch = gaussian_n_components_tf(
            x_channels_tf, y_pred_reshaped_for_reconstruction 
        )
        
        # Calculate MSE between reconstructed spectra
        reconstruction_mse = tf.reduce_sum(tf.square(y_true_reconstructed_spectrum_batch - y_pred_reconstructed_spectrum_batch))
        
        # Total loss combines parameter loss and reconstruction loss
        final_loss = (total_param_loss + reconstruction_mse * reconstruction_weight) / tf.cast(batch_size, tf.float32)

        return final_loss

    return loss_function

def gaussian_n_components(x, *params):
    num_components = len(params) // 3
    if num_components * 3 != len(params):
        raise ValueError("Number of parameters must be a multiple of 3 (amplitude, mean, sigma).")
    y_sum = np.zeros_like(x, dtype=float)
    for i in range(num_components):
        amplitude = params[i * 3]
        mean = params[i * 3 + 1]
        sigma = params[i * 3 + 2]
        if np.isclose(amplitude, 0.0, atol=1e-9):
            continue 
        if np.isclose(sigma, 0.0, atol=1e-9):
            sigma = 1e-9 
        
        y_sum += gaussian(x, amplitude, mean, sigma)
    return y_sum

def _get_active_components(params_flat, max_components, amp_threshold=1e-6, sigma_threshold=1e-6): 
    active_components = []
    for i in range(max_components):
        amp = params_flat[i*3]
        mean = params_flat[i*3 + 1]
        sigma = params_flat[i*3 + 2]
        if amp >= amp_threshold and sigma >= sigma_threshold: 
            active_components.append({'amp': amp, 'mean': mean, 'sigma': sigma, 'original_idx': i})
    return active_components

# --- Function to load model and scalers ---
def _load_model_and_scalers(model_dir):
    """
    Loads the trained Keras model and the associated StandardScaler objects.

    This function now correctly handles the custom_gaussian_loss function
    by passing it to Keras's `custom_objects` during model loading.
    """
    resources = {}
    scalers_to_load = ['scaler_X.pkl', 'scaler_amp.pkl', 'scaler_mean.pkl', 'scaler_sigma.pkl']
    
    for scaler_file in scalers_to_load:
        path = os.path.join(model_dir, scaler_file)
        if not os.path.exists(path):
            print(f"Error: Scaler file not found at {path}")
            return None
        resources[scaler_file.split('.')[0]] = joblib.load(path)

    model_path = os.path.join(model_dir, 'model.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
        
    try:
        # --- THE FIX ---
        # 1. Create the x_channels array that the loss function needs.
        x_channels = np.arange(config.NUM_CHANNELS)
        
        # 2. Define a function wrapper that Keras can use. It has the standard
        #    (y_true, y_pred) signature but calls our real loss with the extra argument.
        def loss_fn_wrapper(y_true, y_pred):
            return custom_gaussian_loss(y_true, y_pred, x_channels)

        # 3. Pass the wrapper in a dictionary to `load_model`. The key must
        #    match the name Keras is looking for, which the error log
        #    tells us is 'loss_function'.
        custom_objects = {'loss_function': loss_fn_wrapper}
        
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        resources['model'] = model
        
        print("✅ Model and scalers loaded successfully.")
        return resources

    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        # This will print the original error, which is helpful for debugging
        import traceback
        traceback.print_exc()
        return None

# --- Plotting Function for ML Predictions on Multiple Examples (MODIFIED) ---
def plot_ml_predictions_on_spectra(
    model, scaler_X, scaler_amp, scaler_mean, scaler_sigma,
    X_data_original, y_data_true_original, metadata_original,
    num_examples: int, x_range: tuple,
    original_indices: np.ndarray, 
    save_dir=None
):
    """
    Plots example spectra with true and ML-predicted Gaussian components overplotted.
    Directly compares components based on their sorted order from data generation.
    """
    print(f"\n--- Plotting {num_examples} example ML predictions (direct fit) ---")
    
    x_channels = np.linspace(x_range[0], x_range[1], config.NUM_CHANNELS) 
    
    num_samples_to_plot = min(num_examples, X_data_original.shape[0])
    sample_indices_in_subset = random.sample(range(X_data_original.shape[0]), num_samples_to_plot)

    print(f"\nGenerating {num_examples} prediction plots...")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Plots will be saved to: {save_dir}")
    
    PLOT_AMP_THRESHOLD = config.PLOT_AMP_THRESHOLD 
    PLOT_SIGMA_THRESHOLD =  config.PLOT_SIGMA_THRESHOLD

    for i, idx_in_subset in enumerate(sample_indices_in_subset):
        original_data_index = original_indices[idx_in_subset]

        observed_spectrum_original = X_data_original[idx_in_subset]
        true_params_flat_original = y_data_true_original[idx_in_subset] 
        spectrum_rms_noise = metadata_original.iloc[idx_in_subset]['rms_noise']

        spectrum_scaled = scaler_X.transform(observed_spectrum_original.reshape(1, -1))
        predicted_params_flat_scaled = model.predict(spectrum_scaled)[0]

        # DEBUG PRINT: Raw SCALED predictions for the first component
        # This will show the model's direct output BEFORE inverse scaling and clamping.
        # It's crucial to see if this is also constant or extremely low/high.
        #print(f"DEBUG: Sample {original_data_index} - Raw SCALED Pred (Amp, Mean, Sigma) for 1st comp:")
        print(f"  Amp: {predicted_params_flat_scaled[0]:.4f}, Mean: {predicted_params_flat_scaled[1]:.4f}, Sigma: {predicted_params_flat_scaled[2]:.4f}")

        predicted_params_flat_original = np.zeros_like(predicted_params_flat_scaled, dtype=float)
        for k_param in range(config.MAX_COMPONENTS_IN_DATA): 
            amp_scaled = predicted_params_flat_scaled[k_param*3]
            mean_scaled = predicted_params_flat_scaled[k_param*3+1]
            sigma_scaled = predicted_params_flat_scaled[k_param*3+2]

            predicted_params_flat_original[k_param*3] = scaler_amp.inverse_transform(np.array([[amp_scaled]]))[0,0]
            predicted_params_flat_original[k_param*3+1] = scaler_mean.inverse_transform(np.array([[mean_scaled]]))[0,0]
            predicted_params_flat_original[k_param*3+2] = scaler_sigma.inverse_transform(np.array([[sigma_scaled]]))[0,0]

        # --- Clamping ALL predicted parameters to their expected ranges ---
        # The variables predicted_params_flat_original[mean_idx] and [sigma_idx] are
        # in INDEX/CHANNEL units after inverse_transform. So they must be clamped
        # with index/channel boundaries defined in config.py.
        for k_param in range(config.MAX_COMPONENTS_IN_DATA): 
            amp_idx = k_param * 3
            mean_idx = k_param * 3 + 1
            sigma_idx = k_param * 3 + 2
            
            # Clamping Amplitude (still using physical bounds, this is correct for amp)
            predicted_params_flat_original[amp_idx] = np.clip(predicted_params_flat_original[amp_idx], config.COMP_AMP_MIN, config.COMP_AMP_MAX)
            
            # Clamping Mean (now using INDEX bounds from config.py)
            predicted_params_flat_original[mean_idx] = np.clip(predicted_params_flat_original[mean_idx], config.ML_PRED_MEAN_INDEX_MIN, config.ML_PRED_MEAN_INDEX_MAX)
            
            # Clamping Sigma (now using CHANNEL bounds from config.py)
            predicted_params_flat_original[sigma_idx] = np.clip(predicted_params_flat_original[sigma_idx], config.ML_PRED_SIGMA_CHAN_MIN, config.ML_PRED_SIGMA_CHAN_MAX)
        # --- END Clamping ALL parameters ---

        y_true_reconstructed_spectrum = gaussian_n_components(x_channels, *true_params_flat_original)
        
        ml_predicted_components_for_sum = []
        for k in range(config.MAX_COMPONENTS_IN_DATA): 
            amp = predicted_params_flat_original[k*3]; mean = predicted_params_flat_original[k*3 + 1]; sigma = predicted_params_flat_original[k*3 + 2]
            if amp >= PLOT_AMP_THRESHOLD and sigma >= PLOT_SIGMA_THRESHOLD:
                ml_predicted_components_for_sum.extend([amp, mean, sigma])
        
        y_ml_predicted_spectrum = np.zeros_like(x_channels)
        if ml_predicted_components_for_sum:
            y_ml_predicted_spectrum = gaussian_n_components(x_channels, *ml_predicted_components_for_sum)

        plt.figure(figsize=(12, 6))
        plt.plot(x_channels, observed_spectrum_original, label='Observed Spectrum', color='gray', alpha=0.7, lw=1)
        plt.plot(x_channels, y_true_reconstructed_spectrum, label='True Underlying Spectrum', color='blue', linestyle='--', lw=2)
        plt.plot(x_channels, y_ml_predicted_spectrum, label=f'ML Predicted Fit (Summed, Filtered)', color='red', lw=2)

        # Plot individual true components
        true_active_count = 0
        for k in range(config.MAX_COMPONENTS_IN_DATA): 
            amp, mean, sigma = true_params_flat_original[k*3:k*3+3]
            if amp >= PLOT_AMP_THRESHOLD and sigma >= PLOT_SIGMA_THRESHOLD:
                true_active_count += 1
                # MODIFIED: Add numerical parameters to the label for true components
                label_text = f'True Comp {k+1} (A:{amp:.3f}, M:{mean:.1f}, S:{sigma:.1f})'
                plt.plot(x_channels, gaussian(x_channels, amp, mean, sigma), color='blue', linestyle=':', alpha=0.6, lw=0.8, label=label_text if true_active_count==1 else "_nolegend_")
                plt.plot(mean, amp, marker='o', color='blue', markersize=8, markeredgewidth=1.5, alpha=0.8, label='_nolegend_')

        # Plot individual predicted components
        pred_active_count = 0
        # DEBUG PRINT: Show the actual predicted parameters for each sample
        #print(f"DEBUG: Sample {original_data_index} - Predicted Parameters (unscaled, clamped):")
        print(f"  Amp: {true_params_flat_original[0]:.4f}, Mean: {true_params_flat_original[1]:.4f}, Sigma: {true_params_flat_original[2]:.4f}")
        # Print all 3*MAX_COMPONENTS_IN_DATA parameters for full diagnostic
        print(f"  {predicted_params_flat_original}") 
        
        for k in range(config.MAX_COMPONENTS_IN_DATA): 
            amp, mean, sigma = predicted_params_flat_original[k*3:k*3+3]
            if amp >= PLOT_AMP_THRESHOLD and sigma >= PLOT_SIGMA_THRESHOLD:
                pred_active_count += 1
                # MODIFIED: Add numerical parameters to the label for predicted components
                label_text = f'Pred Comp {k+1} (A:{amp:.3f}, M:{mean:.1f}, S:{sigma:.1f})'
                plt.plot(x_channels, gaussian(x_channels, amp, mean, sigma), color='red', linestyle=':', alpha=0.6, lw=0.8, label=label_text if pred_active_count==1 else "_nolegend_")
                plt.plot(mean, amp, marker='x', color='red', markersize=8, markeredgewidth=1.5, alpha=0.8, label='_nolegend_')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles)); plt.legend(by_label.values(), by_label.keys())
        plt.title(f'Sample {original_data_index} (RMS: {spectrum_rms_noise:.3f}): True Active {true_active_count} | Pred Active {pred_active_count}')
        plt.xlabel('X (e.g., Velocity)'); plt.ylabel('Intensity'); plt.grid(True); plt.tight_layout()

        # Now, either save or show the plot
        if save_dir:
            plot_filename = os.path.join(save_dir, f'prediction_example_{i+1}.png')
            plt.savefig(plot_filename)
            plt.close() # Important: close the figure to free memory
        else:
            plt.show() # Keep the old behavior if save_dir is not provided

    if save_dir:
        print("Finished saving prediction plots.")

# --- Plotting Function for Multiple Examples from a Saved Model ---
def plot_saved_model_predictions(
    model_dir: str, data_file: str, 
    num_examples: int = 10
):
    """
    Loads a saved Keras model and scalers, then plots example predictions
    from a specified data file without retraining.
    """
    print(f"\n--- Loading model and data for plotting from '{model_dir}' ---")
    
    loaded_resources_common = _load_model_and_scalers(model_dir) 
    if not loaded_resources_common: return

    try:
        store = pd.HDFStore(data_file, mode='r')
        X_spectra_full = store['spectra'].values
        y_true_params_full = np.nan_to_num(store['true_params_ml_format'].values, nan=0.0) 
        metadata_full = store['metadata']
        x_channels_loaded = store['x_channels_template'].values.flatten()
        store.close()
        print(f"Full dataset loaded from {data_file}.")
    except FileNotFoundError as e:
        print(f"Error loading full dataset from {data_file}: {e}"); return
    
    x_range_for_plot_derived = (x_channels_loaded.min(), x_channels_loaded.max())
    print(f"Derived x_range for plotting: {x_range_for_plot_derived}")

    indices = np.arange(X_spectra_full.shape[0])
    from sklearn.model_selection import train_test_split 
    _, X_val, _, y_val, _, idx_val = train_test_split(
        X_spectra_full, y_true_params_full, indices, test_size=0.2, random_state=42
    )
    metadata_val = metadata_full.iloc[idx_val].reset_index(drop=True)

    plot_ml_predictions_on_spectra(
        model=loaded_resources_common['model'], scaler_X=loaded_resources_common['scaler_X'],
        scaler_amp=loaded_resources_common['scaler_amp'], scaler_mean=loaded_resources_common['scaler_mean'], 
        scaler_sigma=loaded_resources_common['scaler_sigma'],
        X_data_original=X_val, y_data_true_original=y_val, metadata_original=metadata_val,
        num_examples=num_examples, x_range=x_range_for_plot_derived,
        original_indices=idx_val
    )
    print(f"\nPlotting for model '{model_dir}' complete.")

# --- Plotting Function for Multiple Examples from a Saved Model ---
# This function is the one to call if you just want to plot predictions from a pre-trained model.
def plot_saved_model_predictions(
    model_dir: str, data_file: str, 
    num_examples: int = 10
):
    """
    Loads a saved Keras model and scalers, then plots example predictions
    from a specified data file without retraining.
    """
    print(f"\n--- Loading model and data for plotting from '{model_dir}' ---")
    
    loaded_resources_common = _load_model_and_scalers(model_dir) 
    if not loaded_resources_common: return

    try:
        store = pd.HDFStore(data_file, mode='r')
        X_spectra_full = store['spectra'].values
        y_true_params_full = np.nan_to_num(store['true_params_ml_format'].values, nan=0.0) 
        metadata_full = store['metadata']
        x_channels_loaded = store['x_channels_template'].values.flatten()
        store.close()
        print(f"Full dataset loaded from {data_file}.")
    except FileNotFoundError as e:
        print(f"Error loading full dataset from {data_file}: {e}"); return
    
    x_range_for_plot_derived = (x_channels_loaded.min(), x_channels_loaded.max())
    print(f"Derived x_range for plotting: {x_range_for_plot_derived}")

    indices = np.arange(X_spectra_full.shape[0])
    # Use sklearn.model_selection.train_test_split directly
    from sklearn.model_selection import train_test_split 
    _, X_val, _, y_val, _, idx_val = train_test_split(
        X_spectra_full, y_true_params_full, indices, test_size=0.2, random_state=42
    )
    metadata_val = metadata_full.iloc[idx_val].reset_index(drop=True)

    plot_ml_predictions_on_spectra(
        model=loaded_resources_common['model'], scaler_X=loaded_resources_common['scaler_X'],
        scaler_amp=loaded_resources_common['scaler_amp'], scaler_mean=loaded_resources_common['scaler_mean'], 
        scaler_sigma=loaded_resources_common['scaler_sigma'],
        X_data_original=X_val, y_data_true_original=y_val, metadata_original=metadata_val,
        num_examples=num_examples, x_range=x_range_for_plot_derived,
        original_indices=idx_val
    )
    print(f"\nPlotting for model '{model_dir}' complete.")