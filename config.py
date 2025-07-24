# config.py

# --- Project Configuration ---
# This file centralizes all major configuration parameters for the HCOP10 project.

# --- General Model and Data Settings ---
# Maximum number of Gaussian components a spectrum can have in generated data.
# This dictates the fixed output size of the ML model (MAX_COMPONENTS_USED_IN_GEN * 3 parameters).
MAX_COMPONENTS_IN_DATA = 2 # For a general multi-component model

# Number of data points (channels) in the spectrum
NUM_CHANNELS = 140

# --- File Paths ---
# Datasets for training (e.g., mix of 0 to MAX_COMPONENTS_IN_DATA)
TRAINING_DATA_FILE = 'spectra_for_ml_training.h5' 
VALIDATION_DATA_FILE = 'spectra_for_ml_validation.h5'

# Directory to save the trained Keras model and its associated scalers
# This directory will be created if it doesn't exist.
MODEL_SAVE_DIR = f'keras_{MAX_COMPONENTS_IN_DATA}_component_model'

# Directory to save plots
PLOT_SAVE_DIR = f'keras_{MAX_COMPONENTS_IN_DATA}_component_model_plots'

# --- Data Generation Script Settings ---
NUM_PLOT_SAMPLE_SPECTRA = 5 # The number of sample spectra plotted after generation
MIN_COMPONENTS_PER_SPECTRUM = 1 # The minimum number of components a spectrum can have
NUM_TRAINING_SPECTRA = 10000 
NUM_VALIDATION_SPECTRA = 1000

# --- Physical Data Ranges (used for data generation and clamping predictions) ---
# These should reflect the expected ranges of your actual HCOP10 spectra.
DATA_X_MIN = 1204.82262682
DATA_X_MAX = 1907.05456744

RMS_NOISE_MIN = 0.0007 # Minimum RMS noise for generated spectra
RMS_NOISE_MAX = 0.0008 # Maximum RMS noise for generated spectra (consider broadening this for realism)

# --- Gaussian Component Parameter Ranges (used for data generation) ---
# These define the range of amplitudes, means, and sigmas for individual Gaussian components.
# Amplitudes should be in the intensity units of your spectra.
COMP_AMP_MIN = 0.002 
COMP_AMP_MAX = 0.05

# Mean positions of components (in X-axis units)
COMP_MEAN_MIN = 1250.0 # Adjusted to match previous discussion
COMP_MEAN_MAX = 1850.0 # Adjusted to match previous discussion

# Sigma (standard deviation) of components (in X-axis units)
COMP_SIGMA_MIN = 5.0
COMP_SIGMA_MAX = 30.0

# --- Clamping Boundaries for ML Output (Index/Channel Units) ---
# These define the strict min/max values for the predicted index/channel parameters.
# They should align with the actual range of indices and channels.

ML_PRED_MEAN_INDEX_MIN = 0.0 # Smallest possible channel index
ML_PRED_MEAN_INDEX_MAX = NUM_CHANNELS - 1.0 # Largest possible channel index (as a float for consistency)

# Calculate min/max sigma in channel units based on physical sigma range and channel width
# Use the overall DATA_X_ range and NUM_CHANNELS to estimate a representative channel width.
# This ensures a consistent channel width for conversion across the spectrum.
_temp_chan_wid_ref = (DATA_X_MAX - DATA_X_MIN) / (NUM_CHANNELS - 1.0) if NUM_CHANNELS > 1 else 1.0 # Use float for NUM_CHANNELS-1

ML_PRED_SIGMA_CHAN_MIN = COMP_SIGMA_MIN / _temp_chan_wid_ref # Convert physical min sigma to channels
ML_PRED_SIGMA_CHAN_MAX = COMP_SIGMA_MAX / _temp_chan_wid_ref # Convert physical max sigma to channels

# It's also often good practice to enforce a hard minimum channel width (e.g., 0.5 or 1.0 channels)
# to avoid unphysical super-narrow peaks if the model predicts very small sigmas.
# This prevents division by zero in downstream calculations if sigma becomes too small.
ML_PRED_SIGMA_CHAN_MIN = max(0.5, ML_PRED_SIGMA_CHAN_MIN) 

# --- ML Model Training Parameters (specific to Keras/TF) ---
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
EPOCHS = 500
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_TOLERANCE = 1e-5

# Thresholds for component matching and plotting (physical values)
# These are used in _get_active_components and plotting functions.
# Set them very low for numerical tolerance for matching
MATCH_AMP_THRESHOLD = 1e-6
MATCH_SIGMA_THRESHOLD = 1e-6

# Plotting thresholds (can be set to ACTUAL_DATA_MINS for visual relevance)
PLOT_AMP_THRESHOLD = RMS_NOISE_MIN
PLOT_SIGMA_THRESHOLD = 1

# --- Plotting Specifics ---
NUM_PLOTS_EXAMPLES = 10 # Number of sample spectra to plot in evaluation

# --- Component Matching Criteria (for match_gaussian_components) ---
# These define how "close" a predicted component must be to a true component to be considered a match.
MATCH_MEAN_DIFF_MAX = 5.0 # Max difference in mean (physical units)
MATCH_SIGMA_DIFF_FACTOR_MAX = 0.5 # Max fractional difference in sigma
MATCH_AMP_DIFF_FACTOR_MAX = 0.8 # Max fractional difference in amplitude