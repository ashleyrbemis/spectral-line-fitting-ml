# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split 
import keras
from keras import layers
from keras.callbacks import EarlyStopping
import joblib
import os
import matplotlib.pyplot as plt
# Make sure to import this Keras utility
from keras.utils import to_categorical

# --- Import Configuration File ---
# Assuming config.py is in the same directory or accessible via sys.path
import config 
from ML_utils import (plot_ml_predictions_on_spectra, 
                      custom_gaussian_loss
                      )


def build_multi_scale_block(input_tensor, filters_per_branch, pool_size=2, dropout_rate=0.2):
    """
    Builds a single Multi-Scale Convolutional Block.
    Takes an input tensor and applies parallel convolutional paths, then concatenates.

    Parameters
    ----------
    input_tensor : tf.Tensor
        The input tensor to this block (e.g., output from a previous layer).
    filters_per_branch : int
        The number of filters for each convolutional branch. Each branch will have this many filters.
    pool_size : int
        The pool_size for MaxPooling after the block (applied to the concatenated output).
    dropout_rate : float
        Dropout rate to apply after the block.

    Returns
    -------
    tf.Tensor
        The output tensor from this multi-scale block.
    """
    # Define branches (e.g., different kernel sizes)
    # Each branch applies Conv1D -> BatchNormalization -> Activation

    # Branch 1: Smaller kernel (e.g., kernel_size=3)
    branch_1 = layers.Conv1D(filters=filters_per_branch, kernel_size=1, padding='same')(input_tensor)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Activation('relu')(branch_1)

    # Branch 2: Medium kernel (e.g., kernel_size=7)
    branch_2 = layers.Conv1D(filters=filters_per_branch, kernel_size=3, padding='same')(input_tensor)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Activation('relu')(branch_2)

    # Branch 3: Larger kernel (e.g., kernel_size=11)
    branch_3 = layers.Conv1D(filters=filters_per_branch, kernel_size=5, padding='same')(input_tensor)
    branch_3 = layers.BatchNormalization()(branch_3)
    branch_3 = layers.Activation('relu')(branch_3)
    
    # You could also add other types of branches, e.g.:
    # Branch 4: Pooling followed by 1x1 convolution (for dimensionality reduction)
    # branch_pool = layers.MaxPooling1D(pool_size=pool_size, padding='same')(input_tensor)
    # branch_pool = layers.Conv1D(filters=filters_per_branch, kernel_size=1, padding='same')(branch_pool) # 1x1 conv to adjust filters
    # branch_pool = layers.BatchNormalization()(branch_pool)
    # branch_pool = layers.Activation('relu')(branch_pool)

    # Concatenate the outputs of the parallel branches
    # The output of concatenation will have a depth equal to sum of filters from all branches
    concatenated_output = layers.concatenate([branch_1, branch_2, branch_3], axis=-1) 
    # Shape of concatenated_output: (batch_size, num_channels_after_conv, filters_per_branch * 3)

    # Apply pooling and dropout after combining features from all scales
    pooled_output = layers.MaxPooling1D(pool_size=pool_size, padding='same')(concatenated_output)
    output_tensor = layers.Dropout(dropout_rate)(pooled_output)

    return output_tensor

# --- Main Training Workflow ---
def run_full_training_workflow():
    
    print(f"Loading training data from {config.TRAINING_DATA_FILE}...") 
    store = pd.HDFStore(config.TRAINING_DATA_FILE, mode='r')

    X_spectra = store['spectra'].values
    y_true_params_raw = store['true_params_ml_format'].values 
    metadata = store['metadata']
    x_channels_loaded = store['x_channels_template'].values.flatten()
    store.close()
    print("Data loaded successfully.")
    
    # Check loaded data dimensions against config.MAX_COMPONENTS_IN_DATA
    if X_spectra.shape[1] != config.NUM_CHANNELS: 
        print(f"Warning: X_spectra has {X_spectra.shape[1]} channels, but expected {config.NUM_CHANNELS}.")
    # y_true_params_raw will have a shape based on MAX_COMPONENTS_IN_DATA (e.g., 3*3=9)
    if y_true_params_raw.shape[1] != config.MAX_COMPONENTS_IN_DATA * 3: 
        print(f"Warning: y_true_params_raw has {y_true_params_raw.shape[1]} output columns, but expected {config.MAX_COMPONENTS_IN_DATA * 3}. This might indicate a mismatch in data generation's padding or config setting.")

    y_true_params = np.nan_to_num(y_true_params_raw, nan=0.0)
    
    indices = np.arange(X_spectra.shape[0])
    
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_spectra, y_true_params, indices, test_size=0.2, random_state=42
    )
    metadata_val = metadata.iloc[idx_val].reset_index(drop=True)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    scaler_amp = StandardScaler()
    scaler_mean = StandardScaler()
    scaler_sigma = StandardScaler()

    # Filters based on numerical zero for active components in Y (labels)
    # This now iterates through all MAX_COMPONENTS_IN_DATA slots for scaling
    active_amps = y_train[:, 0::3][y_train[:, 0::3] > 0.0].reshape(-1, 1) 
    active_means = y_train[:, 1::3][y_train[:, 1::3] > 0.0].reshape(-1, 1) 
    active_sigmas = y_train[:, 2::3][y_train[:, 2::3] > 0.0].reshape(-1, 1) 

    if active_amps.shape[0] > 0: scaler_amp.fit(active_amps)
    else: print("Warning: No active amplitudes in training data. Setting dummy scaler for amp."); scaler_amp.fit(np.array([[0.0],[1.0]]))
    if active_means.shape[0] > 0: scaler_mean.fit(active_means)
    else: print("Warning: No active means in training data. Setting dummy scaler for mean."); scaler_mean.fit(np.array([[0.0],[1.0]]))
    if active_sigmas.shape[0] > 0: scaler_sigma.fit(active_sigmas)
    else: print("Warning: No active sigmas in training data. Setting dummy scaler for sigma."); scaler_sigma.fit(np.array([[0.0],[1.0]]))

    y_train_scaled = np.zeros_like(y_train, dtype=float)
    y_val_scaled = np.zeros_like(y_val, dtype=float)

    # Apply scaling to active components across all MAX_COMPONENTS_IN_DATA slots
    for i_comp in range(config.MAX_COMPONENTS_IN_DATA): 
        active_mask_train = y_train[:, i_comp*3] > 0.0 
        if np.any(active_mask_train):
            y_train_scaled[active_mask_train, i_comp*3] = scaler_amp.transform(y_train[active_mask_train, i_comp*3].reshape(-1, 1)).flatten()
            y_train_scaled[active_mask_train, i_comp*3+1] = scaler_mean.transform(y_train[active_mask_train, i_comp*3+1].reshape(-1, 1)).flatten()
            y_train_scaled[active_mask_train, i_comp*3+2] = scaler_sigma.transform(y_train[active_mask_train, i_comp*3+2].reshape(-1, 1)).flatten()
        
        active_mask_val = y_val[:, i_comp*3] > 0.0
        if np.any(active_mask_val):
            y_val_scaled[active_mask_val, i_comp*3] = scaler_amp.transform(y_val[active_mask_val, i_comp*3].reshape(-1, 1)).flatten()
            y_val_scaled[active_mask_val, i_comp*3+1] = scaler_mean.transform(y_val[active_mask_val, i_comp*3+1].reshape(-1, 1)).flatten()
            y_val_scaled[active_mask_val, i_comp*3+2] = scaler_sigma.transform(y_val[active_mask_val, i_comp*3+2].reshape(-1, 1)).flatten()
        
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True) 
    joblib.dump(scaler_X, os.path.join(config.MODEL_SAVE_DIR, 'scaler_X.pkl'))
    joblib.dump(scaler_amp, os.path.join(config.MODEL_SAVE_DIR, 'scaler_amp.pkl'))
    joblib.dump(scaler_mean, os.path.join(config.MODEL_SAVE_DIR, 'scaler_mean.pkl'))
    joblib.dump(scaler_sigma, os.path.join(config.MODEL_SAVE_DIR, 'scaler_sigma.pkl'))
    print(f"Individual parameter type scalers saved to {config.MODEL_SAVE_DIR}.")

    # --- Keras Model Definition (USING FUNCTIONAL API) ---
    # Define the input layer
    inputs = keras.Input(shape=(config.NUM_CHANNELS,), name='input_spectrum')
    
    # Reshape the input for Conv1D layers
    x = layers.Reshape((config.NUM_CHANNELS, 1), name='reshape_input')(inputs)

    # The total filters will be filters_per_branch * number_of_branches.
    x = build_multi_scale_block(
        input_tensor=x,
        filters_per_branch=32, # Example: 64 filters for each of the 3 branches, total 192 filters
        pool_size=2,
        dropout_rate=0.2
    )

    # --- Flatten and Dense Layers ---
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(512, name='dense_512')(x)
    x = layers.BatchNormalization(name='bn_512')(x)
    x = layers.Activation('relu', name='relu_512')(x)
    x = layers.Dropout(0.3, name='do_512')(x)
    
    x = layers.Dense(256, name='dense_256')(x)
    x = layers.BatchNormalization(name='bn_256')(x)
    x = layers.Activation('relu', name='relu_256')(x)
    x = layers.Dropout(0.3, name='do_256')(x)
    
    outputs = layers.Dense(config.MAX_COMPONENTS_IN_DATA * 3, activation='linear', name='output_layer')(x) 

    # --- Create the Keras Model ---
    model = keras.Model(inputs=inputs, outputs=outputs, name='Gaussian_Fit_Model')
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE) 

    #model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    # Use the custom loss function here, passing all the new weights
    model.compile(
        optimizer=optimizer, 
        loss=custom_gaussian_loss(
            x_channels=x_channels_loaded, # Pass x_channels_loaded
            sparsity_weight=0.01,         # Your current sparsity_weight
            reconstruction_weight=0.1     # Your current reconstruction_weight
        ), 
        metrics=['mae']
    )
    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=config.EARLY_STOPPING_PATIENCE, 
        restore_best_weights=True, verbose=1,
        min_delta=config.EARLY_STOPPING_TOLERANCE 
    )

    print("\nStarting Keras model training for single-component Gaussians...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, 
        validation_data=(X_val_scaled, y_val_scaled),
        callbacks=[early_stopping], verbose=1
    )
    print("Keras model training complete.")

    print("\nEvaluating Keras model on validation set...")
    val_loss, val_mae = model.evaluate(X_val_scaled, y_val_scaled, verbose=0)
    print(f"Keras Model Validation Loss (MSE) on scaled data: {val_loss:.4f}")
    print(f"Keras Model Validation MAE on scaled data: {val_mae:.4f}")

    y_val_pred_scaled = model.predict(X_val_scaled)
    y_val_pred_original_scale = np.zeros_like(y_val_pred_scaled, dtype=float)
    
    for i_comp in range(config.MAX_COMPONENTS_IN_DATA): 
        active_mask_val = y_val[:, i_comp*3] > 0.0 
        if np.any(active_mask_val):
            y_val_pred_original_scale[active_mask_val, i_comp*3] = scaler_amp.inverse_transform(y_val_pred_scaled[active_mask_val, i_comp*3].reshape(-1,1)).flatten()
            y_val_pred_original_scale[active_mask_val, i_comp*3+1] = scaler_mean.inverse_transform(y_val_pred_scaled[active_mask_val, i_comp*3+1].reshape(-1,1)).flatten()
            y_val_pred_original_scale[active_mask_val, i_comp*3+2] = scaler_sigma.inverse_transform(y_val_pred_scaled[active_mask_val, i_comp*3+2].reshape(-1,1)).flatten()

    r2_val_keras = r2_score(y_val, y_val_pred_original_scale)
    print(f"Keras Model Validation R^2 (Scikit-learn, on original scale): {r2_val_keras:.4f}")

    model.save(filepath=os.path.join(config.MODEL_SAVE_DIR, 'model.keras')) 
    print(f"\nKeras model saved to: {config.MODEL_SAVE_DIR}.")

    print("\nPlotting training history...")
    # Ensure the plot directory from config.py exists
    os.makedirs(config.PLOT_SAVE_DIR, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train MSE Loss')
    plt.plot(history.history['val_loss'], label='Validation MSE Loss')
    plt.title('Model MSE Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    
    # Save the figure to the directory specified in config.py
    history_plot_path = os.path.join(config.PLOT_SAVE_DIR, 'training_history.png')
    plt.savefig(history_plot_path)
    plt.close() # Close the figure to free memory
    print(f"Training history plot saved to: {history_plot_path}")

    x_range_for_plot_derived_current_run = (0, len(x_channels_loaded)-1)
    plot_ml_predictions_on_spectra(
        model, scaler_X, scaler_amp, scaler_mean, scaler_sigma, 
        X_val, y_val, metadata_val, 
        num_examples=config.NUM_PLOTS_EXAMPLES, 
        x_range=x_range_for_plot_derived_current_run,
        original_indices=idx_val,
        save_dir=config.PLOT_SAVE_DIR
    )

    print("\nTraining workflow complete. You can now use this model for prediction.")

# --- Main Training Workflow ---
def run_full_training_workflow_not():
    """
    Executes the full workflow for training and evaluating the multi-task
    neural network model for spectral analysis.
    """
    print("--- Starting Multi-Task Training Workflow ---")
    
    # ----------------------------------------------------------------------
    # 1. DATA LOADING
    # ----------------------------------------------------------------------
    print(f"Loading data from {config.TRAINING_DATA_FILE}...")
    try:
        store = pd.HDFStore(config.TRAINING_DATA_FILE, mode='r')
        X_spectra = store['spectra'].values
        y_true_params_raw = store['true_params_ml_format'].values
        metadata = store['metadata']
        x_channels_loaded = store['x_channels_template'].values.flatten()
        store.close()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print("Data loaded successfully.")

    # ----------------------------------------------------------------------
    # 2. LABEL PREPARATION
    # ----------------------------------------------------------------------
    # Regression labels (parameters)
    y_true_params = np.nan_to_num(y_true_params_raw, nan=0.0)

    # Classification labels (component count)
    # Assuming components range from 1 to MAX_COMPONENTS, we map to classes 0 to N-1
    num_classes = config.MAX_COMPONENTS_IN_DATA
    y_true_counts_int = metadata['num_true_components'].values - 1
    y_true_counts_onehot = to_categorical(y_true_counts_int, num_classes=num_classes)
    
    # ----------------------------------------------------------------------
    # 3. DATA SPLITTING
    # ----------------------------------------------------------------------
    indices = np.arange(X_spectra.shape[0])
    X_train, X_val, y_train_params, y_val_params, y_train_counts, y_val_counts, idx_train, idx_val = train_test_split(
        X_spectra, y_true_params, y_true_counts_onehot, indices, test_size=0.2, random_state=42
    )
    metadata_val = metadata.iloc[idx_val].reset_index(drop=True)

    # ----------------------------------------------------------------------
    # 4. DATA SCALING (PREPROCESSING)
    # ----------------------------------------------------------------------
    print("Fitting scalers on training data...")
    # Scale input spectra
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    # Scale output parameters (Amps, Means, Sigmas)
    scaler_amp, scaler_mean, scaler_sigma = StandardScaler(), StandardScaler(), StandardScaler()
    
    y_train_params_scaled = np.zeros_like(y_train_params)
    y_val_params_scaled = np.zeros_like(y_val_params)

    # Fit scalers only on active components in the training set
    for i in range(3): # 0: amp, 1: mean, 2: sigma
        active_params = y_train_params[:, i::3][y_train_params[:, i::3] > 0.0].reshape(-1, 1)
        if active_params.size > 0:
            if i == 0: scaler_amp.fit(active_params)
            elif i == 1: scaler_mean.fit(active_params)
            else: scaler_sigma.fit(active_params)
    
    # Apply fitted scalers to both training and validation sets
    for i_comp in range(config.MAX_COMPONENTS_IN_DATA):
        for scaler, param_idx in zip([scaler_amp, scaler_mean, scaler_sigma], [0, 1, 2]):
            col_idx = i_comp * 3 + param_idx
            
            # Scale training data
            train_mask = y_train_params[:, col_idx] > 0.0
            if np.any(train_mask):
                y_train_params_scaled[train_mask, col_idx] = scaler.transform(y_train_params[train_mask, col_idx].reshape(-1, 1)).flatten()

            # Scale validation data
            val_mask = y_val_params[:, col_idx] > 0.0
            if np.any(val_mask):
                y_val_params_scaled[val_mask, col_idx] = scaler.transform(y_val_params[val_mask, col_idx].reshape(-1, 1)).flatten()

    # Save scalers for later use in prediction
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(config.MODEL_SAVE_DIR, 'scaler_X.pkl'))
    joblib.dump(scaler_amp, os.path.join(config.MODEL_SAVE_DIR, 'scaler_amp.pkl'))
    joblib.dump(scaler_mean, os.path.join(config.MODEL_SAVE_DIR, 'scaler_mean.pkl'))
    joblib.dump(scaler_sigma, os.path.join(config.MODEL_SAVE_DIR, 'scaler_sigma.pkl'))
    print(f"Scalers saved to {config.MODEL_SAVE_DIR}.")

    # ----------------------------------------------------------------------
    # 5. MODEL DEFINITION
    # ----------------------------------------------------------------------
    # Shared backbone
    inputs = keras.Input(shape=(config.NUM_CHANNELS,), name='input_spectrum')
    x = layers.Reshape((config.NUM_CHANNELS, 1))(inputs)
    x = build_multi_scale_block(x, filters_per_branch=64, pool_size=2, dropout_rate=0.25)
    x = build_multi_scale_block(x, filters_per_branch=128, pool_size=2, dropout_rate=0.25)
    x = layers.Flatten()(x)
    shared_dense = layers.Dense(512, activation='relu')(x)
    shared_dense = layers.BatchNormalization()(shared_dense)
    shared_dense = layers.Dropout(0.4)(shared_dense)

    # Output Heads
    # Head 1: Regression for parameters
    param_output = layers.Dense(config.MAX_COMPONENTS_IN_DATA * 3, name='param_output')(shared_dense)
    # Head 2: Classification for component count
    count_output = layers.Dense(num_classes, activation='softmax', name='count_output')(shared_dense)

    model = keras.Model(inputs=inputs, outputs={'param_output': param_output, 'count_output': count_output})
    
    # ----------------------------------------------------------------------
    # 6. MODEL COMPILATION
    # ----------------------------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss={
            'param_output': custom_gaussian_loss(x_channels=x_channels_loaded),
            'count_output': 'categorical_crossentropy'
        },
        loss_weights={'param_output': 1.0, 'count_output': 0.5},
        metrics={'param_output': 'mae', 'count_output': 'accuracy'}
    )
    model.summary()

    # ----------------------------------------------------------------------
    # 7. MODEL TRAINING
    # ----------------------------------------------------------------------
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    y_train_dict = {'param_output': y_train_params_scaled, 'count_output': y_train_counts}
    y_val_dict = {'param_output': y_val_params_scaled, 'count_output': y_val_counts}

    print("\nStarting model training...")
    history = model.fit(
        X_train_scaled, y_train_dict,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val_scaled, y_val_dict),
        callbacks=[early_stopping],
        verbose=1
    )
    print("Model training complete.")
    
    # ----------------------------------------------------------------------
    # 8. MODEL EVALUATION
    # ----------------------------------------------------------------------
    print("\nEvaluating model on validation set...")
    # `model.predict` returns a dictionary with predictions from each head
    predictions = model.predict(X_val_scaled)
    y_val_pred_params_scaled = predictions['param_output']
    y_val_pred_counts_proba = predictions['count_output']

    # --- Evaluate Regression Performance ---
    # Inverse transform parameters to original scale for meaningful R^2 score
    y_val_pred_params_orig = np.zeros_like(y_val_pred_params_scaled)
    for i_comp in range(config.MAX_COMPONENTS_IN_DATA):
        for scaler, param_idx in zip([scaler_amp, scaler_mean, scaler_sigma], [0, 1, 2]):
            col_idx = i_comp * 3 + param_idx
            val_mask = y_val_params[:, col_idx] > 0.0 # Use true mask for evaluation
            if np.any(val_mask):
                y_val_pred_params_orig[val_mask, col_idx] = scaler.inverse_transform(y_val_pred_params_scaled[val_mask, col_idx].reshape(-1, 1)).flatten()
    
    r2_val = r2_score(y_val_params[y_val_params > 0], y_val_pred_params_orig[y_val_params > 0])
    print(f"\nParameter Prediction (Regression) R^2 Score: {r2_val:.4f}")

    # --- Evaluate Classification Performance ---
    y_val_pred_counts_int = np.argmax(y_val_pred_counts_proba, axis=1)
    y_val_true_counts_int = np.argmax(y_val_counts, axis=1) # Convert one-hot back to int
    
    print("\nComponent Count Prediction (Classification) Report:")
    print(classification_report(y_val_true_counts_int, y_val_pred_counts_int))
    
    # ----------------------------------------------------------------------
    # 9. SAVING & PLOTTING
    # ----------------------------------------------------------------------
    model.save(os.path.join(config.MODEL_SAVE_DIR, 'multi_task_model.keras'))
    print(f"\nModel saved to: {config.MODEL_SAVE_DIR}")

    # Ensure the plot directory from config.py exists
    os.makedirs(config.PLOT_SAVE_DIR, exist_ok=True)

    # Plot training history
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Total Train Loss')
    plt.plot(history.history['val_loss'], label='Total Validation Loss')
    plt.title('Total Model Loss'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['count_output_accuracy'], label='Train Accuracy (Count)')
    plt.plot(history.history['val_count_output_accuracy'], label='Validation Accuracy (Count)')
    plt.title('Component Count Accuracy'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.tight_layout()

    # Save the figure
    history_plot_path_mt = os.path.join(config.PLOT_SAVE_DIR, 'multi_task_training_history.png')
    plt.savefig(history_plot_path_mt)
    plt.close()
    print(f"Multi-task training history plot saved to: {history_plot_path_mt}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_val_true_counts_int, y_val_pred_counts_int)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(1, num_classes + 1), 
                yticklabels=range(1, num_classes + 1))
    plt.title('Confusion Matrix for Component Count'); plt.ylabel('True Count'); plt.xlabel('Predicted Count')

    # Save the figure
    cm_plot_path = os.path.join(config.PLOT_SAVE_DIR, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_plot_path}")

    # Plot spectral fit examples (using your existing utility)
    # Note: The utility will need to be adapted to accept the separate scalers
    # and handle the multi-output model. This is a placeholder for that call.
    #print("\nVisualizing predictions on sample spectra...")
    # plot_ml_predictions_on_spectra(...) # This would require modification

    print("\n--- Full Training Workflow Finished ---")


if __name__ == '__main__':
    run_full_training_workflow()
# %%
