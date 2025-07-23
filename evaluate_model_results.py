# %%
# evaluate_model_results.py
# This script loads a trained Keras model and its associated scalers,
# loads a validation dataset, evaluates the model's performance,
# and visualizes sample fit results.

import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# --- Set Path for Module Imports ---
script_dir = os.path.dirname(__file__) 
if script_dir not in sys.path:
    sys.path.append(script_dir)

from ML_utils import (_load_model_and_scalers, 
                      plot_ml_predictions_on_spectra
                      )

# Import configuration
import config

# --- Main execution block for evaluation ---
if __name__ == '__main__':
    print("🚀 Starting Model Evaluation Script 🚀")
    
    # --- Load Model and Scalers ---
    loaded_resources = _load_model_and_scalers(config.MODEL_SAVE_DIR)
    if not loaded_resources:
        print("🔴 Failed to load model or scalers. Exiting.")
        exit()
    
    model = loaded_resources['model']
    scaler_X = loaded_resources['scaler_X']
    scaler_amp = loaded_resources['scaler_amp']
    scaler_mean = loaded_resources['scaler_mean']
    scaler_sigma = loaded_resources['scaler_sigma']

    # --- Load Validation Data ---
    print(f"\nLoading validation data from: {config.VALIDATION_DATA_FILE}")
    try:
        store_val = pd.HDFStore(config.VALIDATION_DATA_FILE, mode='r')
        X_val_spectra = store_val['spectra'].values
        y_val_params_raw = store_val['true_params_ml_format'].values 
        y_val_params_processed = np.nan_to_num(y_val_params_raw, nan=0.0)
        
        metadata_val = store_val['metadata']
        x_channels_template_loaded = store_val['x_channels_template'].values.flatten()
        store_val.close()
        print("Validation data loaded successfully.")
    except FileNotFoundError as e:
        print(f"🔴 Error: Validation data file not found at {config.VALIDATION_DATA_FILE}. Please ensure it exists.")
        exit()
    except Exception as e:
        print(f"🔴 Error loading validation data: {e}")
        exit()
        
    print(f"Validation Data - X_spectra shape: {X_val_spectra.shape}")
    print(f"Validation Data - y_params_processed shape: {y_val_params_processed.shape}")

    # Scale validation data using the loaded scaler_X
    X_val_scaled = scaler_X.transform(X_val_spectra)

    # --- Make Predictions ---
    print("\nMaking predictions on the validation set...")
    y_val_pred_scaled = model.predict(X_val_scaled)

    # Inverse transform predictions to get back to original parameter units
    y_val_pred_original_units = np.zeros_like(y_val_pred_scaled, dtype=float)
    # Loop through config.MAX_COMPONENTS_IN_DATA to process all potential component slots
    for i_comp in range(config.MAX_COMPONENTS_IN_DATA ): # Correctly uses config.MAX_COMPONENTS_IN_DATA 
        y_val_pred_original_units[:, i_comp*3] = scaler_amp.inverse_transform(y_val_pred_scaled[:, i_comp*3].reshape(-1,1)).flatten()
        y_val_pred_original_units[:, i_comp*3+1] = scaler_mean.inverse_transform(y_val_pred_scaled[:, i_comp*3+1].reshape(-1,1)).flatten()
        y_val_pred_original_units[:, i_comp*3+2] = scaler_sigma.inverse_transform(y_val_pred_scaled[:, i_comp*3+2].reshape(-1,1)).flatten()

    # --- Evaluate Model Performance ---
    print("\n--- Model Performance Metrics ---")
    
    # This metric compares the full arrays, including padded zeros, which can be misleading.
    overall_r2 = r2_score(y_val_params_processed, y_val_pred_original_units)
    print(f"Overall R^2 (all parameters, including padding): {overall_r2:.4f}")

    # An active component is one where the true parameters are not zero.
    active_mask = y_val_params_processed > 1e-9 # Use a small threshold to avoid float precision issues
    
    if np.any(active_mask):
        r2_active_params = r2_score(y_val_params_processed[active_mask], y_val_pred_original_units[active_mask])
        mse_active_params = mean_squared_error(y_val_params_processed[active_mask], y_val_pred_original_units[active_mask])
        print(f"\nMetrics on Active (non-zero) Parameters Only:")
        print(f"  R^2 (Active Parameters): {r2_active_params:.4f}")
        print(f"  MSE (Active Parameters): {mse_active_params:.6f}")

        print("\nPerformance Breakdown by Parameter Type (Active Components):")
        param_names = ['Amplitude', 'Mean (Index)', 'Sigma (Channels)']
        for i, param_name in enumerate(param_names):
            true_params = y_val_params_processed[:, i::3]
            pred_params = y_val_pred_original_units[:, i::3]
            
            # Create a mask for just this parameter type
            param_active_mask = true_params > 1e-9
            
            if np.any(param_active_mask):
                param_r2 = r2_score(true_params[param_active_mask], pred_params[param_active_mask])
                param_mse = mean_squared_error(true_params[param_active_mask], pred_params[param_active_mask])
                print(f"  {param_name} - R2: {param_r2:.4f}, MSE: {param_mse:.6f}")
            else:
                print(f"  No active components found for parameter: {param_name}")
    else:
        print("No active components found in the validation set for detailed evaluation.")

    ## Calculate metrics specifically for N-component spectra to match your primary goal
    #one_comp_val_indices = metadata_val[metadata_val['num_true_components'] == 1].index
    #if len(one_comp_val_indices) > 0:
    #    # Use config.MAX_COMPONENTS_IN_DATA for slicing y_val_params_processed, then take [:3]
    #    # This correctly handles the padding.
    #    y_true_one_comp = y_val_params_processed[one_comp_val_indices][:, :3] 
    #    y_pred_one_comp = y_val_pred_original_units[one_comp_val_indices][:, :3]
    #
    #    mse_one_comp = mean_squared_error(y_true_one_comp, y_pred_one_comp)
    #    r2_one_comp = r2_score(y_true_one_comp, y_pred_one_comp)
    #    print(f"R^2: {r2_one_comp:.4f}")
    #    print(f"MSE: {mse_one_comp:.6f}")
    #
    #    param_names = ['Amplitude', 'Mean (Index)', 'Sigma (Channels)']
    #    for i, param_name in enumerate(param_names):
    #        param_mse = mean_squared_error(y_true_one_comp[:, i], y_pred_one_comp[:, i])
    #        param_r2 = r2_score(y_true_one_comp[:, i], y_pred_one_comp[:, i])
    #        print(f"  {param_name} - MSE: {param_mse:.6f}, R2: {param_r2:.4f}")
    #else:
    #    print("No 1-component spectra found in the validation set for specific evaluation.")

    # --- NEW: Evaluate Component Count Detection ---
    print("\n--- Component Count Detection Metrics ---")
    
    # Define a threshold for what counts as a 'detected' component
    # Using the plot threshold from config is a sensible, physically-motivated choice
    detection_threshold = config.PLOT_AMP_THRESHOLD 
    
    # Get the true number of components from metadata
    true_counts = metadata_val['num_true_components'].values
    
    # Calculate the predicted number of components based on the threshold
    predicted_amplitudes = y_val_pred_original_units[:, 0::3]
    predicted_counts = np.sum(predicted_amplitudes > detection_threshold, axis=1)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(true_counts, predicted_counts)
    print(f"Component Count Accuracy: {accuracy:.4f}")
    
    # Generate and save a confusion matrix
    os.makedirs(config.PLOT_SAVE_DIR, exist_ok=True)
    cm = confusion_matrix(true_counts, predicted_counts)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(true_counts), 
                yticklabels=np.unique(true_counts))
    plt.title('Confusion Matrix for Component Count')
    plt.ylabel('True Count')
    plt.xlabel('Predicted Count')
    
    cm_plot_path = os.path.join(config.PLOT_SAVE_DIR, 'component_count_confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_plot_path}")

    # --- Visualize Sample Predictions ---
    print("\n--- Visualizing Sample Predictions ---")

    # Ensure the plot directory from config.py exists
    os.makedirs(config.PLOT_SAVE_DIR, exist_ok=True)
    
    dummy_val_indices = np.arange(X_val_spectra.shape[0])

    plot_ml_predictions_on_spectra(
        model=model, scaler_X=scaler_X, scaler_amp=scaler_amp, scaler_mean=scaler_mean,
        scaler_sigma=scaler_sigma,
        X_data_original=X_val_spectra, 
        y_data_true_original=y_val_params_processed, 
        metadata_original=metadata_val, 
        num_examples=config.NUM_PLOTS_EXAMPLES, # Use from config.py
        x_range=(0, config.NUM_CHANNELS-1),
        original_indices=dummy_val_indices,
        save_dir=config.PLOT_SAVE_DIR
    )
    print(f"\nEvaluation complete. Check '{config.PLOT_SAVE_DIR}' for visual assessment.")