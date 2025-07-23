# %%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os
import random
import matplotlib.pyplot as plt

# --- Configuration ---
ML_TRAINING_DATA_FILE = 'hcop10_ml_training_spectra.h5' # The HDF5 file you generated
OUTPUT_ML_MODEL_FILE = 'hcop10_gaussian_param_predictor.pkl' # Where to save the trained model

# These should match what you used when generating the data
MAX_COMPONENTS_USED_IN_GEN = 3
NUM_CHANNELS_USED_IN_GEN = 140

def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-((x - mean)**2) / (2 * sigma**2))

def gaussian_n_components(x, *params):
    num_components = len(params) // 3
    if num_components * 3 != len(params):
        raise ValueError("Number of parameters must be a multiple of 3 (amplitude, mean, sigma).")
    y_sum = np.zeros_like(x, dtype=float)
    for i in range(num_components):
        amplitude = params[i * 3]
        mean = params[i * 3 + 1]
        sigma = params[i * 3 + 2]
        if sigma > 0 and amplitude >= 0:
            y_sum += gaussian(x, amplitude, mean, sigma)
    return y_sum

def _get_active_components(params_flat, max_components, amp_threshold=0.01, sigma_threshold=0.01):
    active_components = []
    for i in range(max_components):
        amp = params_flat[i*3]
        mean = params_flat[i*3 + 1]
        sigma = params_flat[i*3 + 2]
        if amp > amp_threshold and sigma > sigma_threshold:
            active_components.append({
                'amp': amp, 'mean': mean, 'sigma': sigma, 'original_idx': i
            })
    return active_components

def match_gaussian_components(
    true_params_flat: np.ndarray, predicted_params_flat: np.ndarray, max_components: int,
    amp_match_threshold: float = 0.05, sigma_match_threshold: float = 0.05,
    max_mean_diff: float = 5.0, max_sigma_diff_factor: float = 0.5, max_amp_diff_factor: float = 0.8
) -> dict:
    true_comps = _get_active_components(true_params_flat, max_components, amp_match_threshold, sigma_match_threshold)
    pred_comps = _get_active_components(predicted_params_flat, max_components, amp_match_threshold, sigma_match_threshold)

    num_true, num_pred = len(true_comps), len(pred_comps)
    if num_true == 0 and num_pred == 0: return {'matches': [], 'missed_true_idx': [], 'spurious_pred_idx': []}
    if num_true == 0: return {'matches': [], 'missed_true_idx': [], 'spurious_pred_idx': [c['original_idx'] for c in pred_comps]}
    if num_pred == 0: return {'matches': [], 'missed_true_idx': [c['original_idx'] for c in true_comps], 'spurious_pred_idx': []}

    cost_matrix = np.full((num_true, num_pred), np.inf)
    for i in range(num_true):
        t_comp = true_comps[i]
        for j in range(num_pred):
            p_comp = pred_comps[j]
            mean_diff = np.abs(t_comp['mean'] - p_comp['mean'])
            sigma_diff_frac = np.abs(t_comp['sigma'] - p_comp['sigma']) / (0.5 * (t_comp['sigma'] + p_comp['sigma']) + 1e-9)
            amp_diff_frac = np.abs(t_comp['amp'] - p_comp['amp']) / (0.5 * (t_comp['amp'] + p_comp['amp']) + 1e-9)
            if (mean_diff <= max_mean_diff and sigma_diff_frac <= max_sigma_diff_factor and amp_diff_frac <= max_amp_diff_factor):
                cost = (mean_diff / max_mean_diff) + (sigma_diff_frac / max_sigma_diff_factor) + (amp_diff_frac / max_amp_diff_factor)
                cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    matched_true_indices = set()
    matched_pred_indices = set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] != np.inf:
            matches.append((true_comps[r]['original_idx'], pred_comps[c]['original_idx']))
            matched_true_indices.add(true_comps[r]['original_idx'])
            matched_pred_indices.add(pred_comps[c]['original_idx'])
    missed_true_idx = [t['original_idx'] for t in true_comps if t['original_idx'] not in matched_true_indices]
    spurious_pred_idx = [p['original_idx'] for p in pred_comps if p['original_idx'] not in matched_pred_indices]
    return {'matches': matches, 'missed_true_idx': missed_true_idx, 'spurious_pred_idx': spurious_pred_idx}

# --- NEW FUNCTION: plot_rf_predictions_on_spectra_from_file ---
def plot_rf_predictions_on_spectra_from_file(
    model_file_path: str,
    scaler_X_path: str,
    scaler_y_path: str,
    data_file: str,
    num_examples: int = 10,
    amp_threshold_for_summed_fit: float = 0.1,
    # Match criteria for component plots
    match_max_mean_diff: float = 5.0, # km/s, example tolerance
    match_max_sigma_diff_factor: float = 0.5, # 50% diff
    match_max_amp_diff_factor: float = 0.8 # 80% diff
):
    """
    Loads a saved RandomForestRegressor model and its scalers, then plots
    example predictions from the validation set of the specified HDF5 data.
    """
    print(f"\n--- Loading RF model and data for plotting from '{model_file_path}' ---")

    # 1. Load the RandomForestRegressor model
    try:
        model = joblib.load(model_file_path)
        print(f"RandomForestRegressor model loaded from {model_file_path}")
    except Exception as e:
        print(f"Error loading RandomForestRegressor model from {model_file_path}: {e}")
        return

    # 2. Load the X (input) and Y (output) scalers
    try:
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        print(f"Scalers loaded from {scaler_X_path} and {scaler_y_path}")
    except FileNotFoundError as e:
        print(f"Error loading scaler files: {e}. Make sure they exist.")
        return
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return

    # 3. Load the full dataset (for validation set and true x_range)
    try:
        store = pd.HDFStore(data_file, mode='r')
        X_spectra_full = store['spectra'].values
        y_true_params_full = store['true_params'].values
        metadata_full = store['metadata']
        x_channels_loaded = store['x_channels'].values.flatten()
        store.close()
        print(f"Full dataset loaded from {data_file}.")
    except FileNotFoundError as e:
        print(f"Error loading full dataset from {data_file}: {e}")
        return
    except Exception as e:
        print(f"Error reading HDF5 data: {e}")
        return

    # 4. Derive x_range from loaded x_channels
    x_range_for_plot_derived = (x_channels_loaded.min(), x_channels_loaded.max())
    print(f"Derived x_range for plotting: {x_range_for_plot_derived}")

    # 5. Perform the exact same train_test_split to get the validation set and original indices
    indices = np.arange(X_spectra_full.shape[0])
    _, X_val, _, y_val, _, idx_val = train_test_split(
        X_spectra_full, y_true_params_full, indices, test_size=0.2, random_state=42
    )
    metadata_val = metadata_full.iloc[idx_val].reset_index(drop=True)
    print(f"Validation set split: {X_val.shape[0]} samples.")

    # 6. Plotting loop
    print(f"\n--- Plotting {num_examples} example RF predictions with component matching ---")
    x_plot_channels = np.linspace(x_range_for_plot_derived[0], x_range_for_plot_derived[1], NUM_CHANNELS_USED_IN_GEN)
    
    sample_indices_in_subset = random.sample(range(X_val.shape[0]), num_examples)

    # Global constants for clamping (should match data generation's x_range or your desired physical bounds)
    # These might need to be adjusted based on your generated data's actual x_range
    ACTUAL_DATA_X_MIN = x_range_for_plot_derived[0] # Clamp to the range of the loaded data
    ACTUAL_DATA_X_MAX = x_range_for_plot_derived[1]

    for i, idx_in_subset in enumerate(sample_indices_in_subset):
        original_data_index = idx_val[idx_in_subset]

        observed_spectrum_original = X_val[idx_in_subset]
        true_params_flat_original = y_val[idx_in_subset]
        spectrum_rms_noise = metadata_val.iloc[idx_in_subset]['rms_noise']

        # Scale the input spectrum for prediction
        spectrum_scaled = scaler_X.transform(observed_spectrum_original.reshape(1, -1))
        
        # Make prediction
        predicted_params_flat_original_raw = model.predict(spectrum_scaled)[0] # RandomForest outputs original scale directly if not using pipelines
        
        # --- Clamping predicted means to the expected range (as discussed) ---
        # RandomForestRegressor can sometimes predict values outside the training range,
        # especially if the target distributions are wide or if it encounters unusual feature combinations.
        predicted_params_clamped = np.copy(predicted_params_flat_original_raw) # Work on a copy
        for k_param in range(MAX_COMPONENTS_USED_IN_GEN):
            mean_idx = k_param * 3 + 1
            predicted_params_clamped[mean_idx] = np.clip(
                predicted_params_clamped[mean_idx],
                ACTUAL_DATA_X_MIN,
                ACTUAL_DATA_X_MAX
            )
            # Optional: Clamp amplitudes and sigmas too if they can go negative/non-physical
            # amp_idx = k_param * 3
            # predicted_params_clamped[amp_idx] = np.clip(predicted_params_clamped[amp_idx], 0.0, np.inf)
            # sigma_idx = k_param * 3 + 2
            # predicted_params_clamped[sigma_idx] = np.clip(predicted_params_clamped[sigma_idx], 0.0, np.inf)
        # Use the clamped parameters for matching and plotting
        predicted_params_final = predicted_params_clamped


        # --- Perform Component Matching ---
        matching_results = match_gaussian_components(
            true_params_flat_original,
            predicted_params_final, # Use clamped predictions for matching
            MAX_COMPONENTS_USED_IN_GEN,
            max_mean_diff=match_max_mean_diff,
            max_sigma_diff_factor=match_max_sigma_diff_factor,
            max_amp_diff_factor=match_max_amp_diff_factor
        )
        matches, missed_true_idx, spurious_pred_idx = matching_results['matches'], matching_results['missed_true_idx'], matching_results['spurious_pred_idx']

        # --- Prepare spectra for plotting ---
        y_true_reconstructed_spectrum = gaussian_n_components(x_plot_channels, *true_params_flat_original)
        
        ml_predicted_components_for_sum = []
        for k in range(MAX_COMPONENTS_USED_IN_GEN):
            amp = predicted_params_final[k*3]; mean = predicted_params_final[k*3 + 1]; sigma = predicted_params_final[k*3 + 2]
            if amp > amp_threshold_for_summed_fit and sigma > 0:
                ml_predicted_components_for_sum.extend([amp, mean, sigma])
        
        if not ml_predicted_components_for_sum:
            y_ml_predicted_spectrum = np.zeros_like(x_plot_channels)
        else:
            y_ml_predicted_spectrum = gaussian_n_components(x_plot_channels, *ml_predicted_components_for_sum)


        # --- Create figure and plot ---
        plt.figure(figsize=(12, 6))
        
        plt.plot(x_plot_channels, observed_spectrum_original, label='Observed Spectrum', color='gray', alpha=0.7, lw=1)
        plt.plot(x_plot_channels, y_true_reconstructed_spectrum, label='True Underlying Spectrum', color='blue', linestyle='--', lw=2)
        plt.plot(x_plot_channels, y_ml_predicted_spectrum, label=f'ML Predicted Fit (Summed, Filtered)', color='red', lw=2)


        # Plot individual TRUE components (distinguish matched/missed)
        true_raw_count = 0
        for k in range(MAX_COMPONENTS_USED_IN_GEN):
            amp, mean, sigma = true_params_flat_original[k*3:k*3+3]
            if amp > 0.001 and sigma > 0.001:
                true_raw_count += 1
                is_matched_true = any(k == m[0] for m in matches)
                color = 'blue' if is_matched_true else ('red' if k in missed_true_idx else 'blue')
                marker = 'o' if is_matched_true else ('x' if k in missed_true_idx else 's')
                m_label = 'True Matched' if is_matched_true else ('True Missed' if k in missed_true_idx else '_nolegend_')
                
                plt.plot(x_plot_channels, gaussian(x_plot_channels, amp, mean, sigma), color=color, linestyle=':', alpha=0.6, lw=0.8, label=m_label if not m_label == '_nolegend_' else None)
                plt.plot(mean, amp, marker=marker, color=color, markersize=8, markeredgewidth=1.5, alpha=0.8, label='_nolegend_')


        # Plot individual ML predicted components (distinguish matched/spurious)
        pred_raw_count = 0
        for k in range(MAX_COMPONENTS_USED_IN_GEN):
            amp, mean, sigma = predicted_params_final[k*3:k*3+3] # Use final clamped predictions
            if amp > 0.001 and sigma > 0.001:
                pred_raw_count += 1
                is_matched_pred = any(k == m[1] for m in matches)
                color = 'green' if is_matched_pred else ('purple' if k in spurious_pred_idx else 'orange')
                marker = '^' if is_matched_pred else ('v' if k in spurious_pred_idx else '+')
                m_label = 'Pred Matched' if is_matched_pred else ('Pred Spurious' if k in spurious_pred_idx else '_nolegend_')
                
                plt.plot(x_plot_channels, gaussian(x_plot_channels, amp, mean, sigma), color=color, linestyle=':', alpha=0.6, lw=0.8, label=m_label if not m_label == '_nolegend_' else None)
                plt.plot(mean, amp, marker=marker, color=color, markersize=8, markeredgewidth=1.5, alpha=0.8, label='_nolegend_')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())


        plt.title(f'Sample {original_data_index} (RMS: {spectrum_rms_noise:.2f}): True {true_raw_count} | Pred {pred_raw_count} Raw | Match {len(matches)} | Miss {len(missed_true_idx)} | Spurious {len(spurious_pred_idx)}')
        plt.xlabel('X (e.g., Velocity)')
        plt.ylabel('Intensity')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print("\nExample RF plots displayed.")

def generate_training_data():
    # --- 1. Load the Generated Training Data ---
    print(f"Loading training data from {ML_TRAINING_DATA_FILE}...")
    try:
        store = pd.HDFStore(ML_TRAINING_DATA_FILE, mode='r')
        X_spectra = store['spectra'].values # Your observed spectra (features)
        y_true_params = store['true_params'].values # Your true Gaussian parameters (targets)
        metadata = store['metadata'] # Metadata like num_true_components, rms_noise
        # x_channels = store['x_channels'].values.flatten() # Not strictly needed for training, but good to have
        store.close()
        print("Data loaded successfully.")
        print(f"X_spectra shape: {X_spectra.shape}")
        print(f"y_true_params shape: {y_true_params.shape}")
        print(f"Metadata shape: {metadata.shape}")

        # Verify consistency (optional but good practice)
        if X_spectra.shape[1] != NUM_CHANNELS_USED_IN_GEN:
            print(f"Warning: X_spectra has {X_spectra.shape[1]} channels, but expected {NUM_CHANNELS_USED_IN_GEN}.")
        if y_true_params.shape[1] != MAX_COMPONENTS_USED_IN_GEN * 3:
            print(f"Warning: y_true_params has {y_true_params.shape[1]} output columns, but expected {MAX_COMPONENTS_USED_IN_GEN * 3}.")

    except FileNotFoundError:
        print(f"Error: Training data file '{ML_TRAINING_DATA_FILE}' not found.")
        print("Please run `generate_ml_training_data()` first to create it.")
        exit() # Exit if data is not found

    # --- 2. Data Preprocessing and Splitting ---

    # Handle NaNs if they were used for padding in y_true_params
    # For regression, it's often better to train a separate model per component,
    # or for a fixed number of components and then filter.
    # If you padded with NaNs during generation, you might need to convert them to zeros,
    # or mask them, or use a model that handles NaNs.
    # Assuming here that you either used 0.0 for padding or your regressor handles NaNs gracefully
    # (RandomForestRegressor can often handle NaNs by treating them as a separate category,
    # but it's better to explicitly handle them for numerical regression).
    # If you used np.nan for padding in generate_ml_training_data, it's safer to convert them to 0.
    # The ML model will learn that 0-amplitude components are "missing".
    y_true_params[np.isnan(y_true_params)] = 0.0


    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_spectra, y_true_params, test_size=0.2, random_state=42 # 20% for validation
    )

    print(f"\nTraining set size: {X_train.shape[0]} spectra")
    print(f"Validation set size: {X_val.shape[0]} spectra")

    # --- 3. Define and Train the Machine Learning Pipeline ---

    # The pipeline combines scaling and the regressor
    ml_pipeline = Pipeline([
        ('scaler', StandardScaler()), # Scales each spectral channel (feature)
        ('regressor', RandomForestRegressor(
            n_estimators=100,      # Number of trees in the forest
            random_state=42,       # For reproducibility
            n_jobs=-1,             # Use all available CPU cores for faster training
            max_depth=15,           # Limit tree depth to prevent extreme overfitting (optional)
            verbose=1,
        ))
        # Alternative for a Neural Network (MLP):
        # ('regressor', MLPRegressor(
        #     hidden_layer_sizes=(128, 64), # Two hidden layers with 128 and 64 neurons
        #     activation='relu',           # Rectified Linear Unit activation
        #     solver='adam',               # Optimization algorithm
        #     max_iter=500,                # Max iterations for solver
        #     random_state=42,
        #     verbose=True                 # Show training progress
        # ))
    ])


    print("\nStarting ML model training...")
    ml_pipeline.fit(X_train, y_train)
    print("ML model training complete.")

    # --- 4. Evaluate the Trained Model ---
    train_score = ml_pipeline.score(X_train, y_train)
    val_score = ml_pipeline.score(X_val, y_val) # R^2 score for regression

    print(f"\nML Model Training R^2 score: {train_score:.4f}")
    print(f"ML Model Validation R^2 score: {val_score:.4f}")

    # --- 5. Save the Trained ML Model ---
    joblib.dump(ml_pipeline, OUTPUT_ML_MODEL_FILE)
    print(f"\nML model saved to: {OUTPUT_ML_MODEL_FILE}")

    print("\nTraining workflow complete. You can now use this model for prediction.")

# --- Main execution block (example usage for the new function) ---
if __name__ == '__main__':
    # To run this, ensure your RandomForestRegressor training (the long one) has completed
    # and saved its model to 'hcop10_gaussian_param_predictor.pkl'
    # and its scalers to 'scaler_X_for_keras_model.pkl' and 'scaler_y_for_keras_model.pkl'

    generate_training_data()

    #print("\n--- Running RandomForestRegressor plotting example ---")
    #plot_rf_predictions_on_spectra_from_file(
    #    model_file_path=OUTPUT_ML_MODEL_FILE, # This points to the .pkl file
    #    scaler_X_path='scaler_X_for_keras_model.pkl',
    #    scaler_y_path='scaler_y_for_keras_model.pkl',
    #    data_file=ML_TRAINING_DATA_FILE, # This is 'hcop10_ml_training_spectra.h5'
    #    num_examples=10 # Plot 10 random examples
    #)