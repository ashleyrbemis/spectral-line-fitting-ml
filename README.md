# Machine Learning for Spectral Line Fitting

## 📖 Overview

This project implements a deep learning model (a Convolutional Neural Network) to analyze spectral data and automatically identify the number and parameters of underlying Gaussian components. Traditional fitting methods can be slow and require good initial guesses, whereas this ML approach aims to provide a fast and accurate initial analysis of complex spectra.

**Problem Statement:** Given a noisy one-dimensional spectrum, the goal is to predict the parameters (Amplitude, Mean, Sigma) for each Gaussian component present.

---

## ✨ Key Features & Implementation Details

* **Synthetic Data Generation:** Includes a data pipeline to generate large, labeled datasets of synthetic spectra. This enables controlled training and validation without reliance on scarce, real-world unlabeled data.

* **Custom CNN Architecture:** The model, built in TensorFlow/Keras, uses custom multi-scale convolutional blocks. These blocks apply parallel convolutions with different kernel sizes to simultaneously learn both broad and sharp spectral features, improving parameter extraction.

* **Data Preprocessing & Validation :** Implemented a robust data preprocessing workflow using scikit-learn. This involved feature scaling on the input spectra with StandardScaler to normalize the data, and partitioning the dataset into training and validation sets to prevent data leakage and ensure a fair evaluation of model performance.

* **Supervised Training & Model Architecture:** Batch normalization is used throughout the network to stabilize and accelerate training. The model is trained via supervised learning, mapping an input spectrum to its corresponding ground truth labels: the amplitude, mean, and standard deviation (sigma) for each underlying Gaussian component.

* **Custom Multi-Part Loss Function:** A key feature is the custom loss function that combines three weighted objectives to guide the model:
    * **Parameter MSE:** Penalizes errors in the predicted parameters (amplitude, mean, sigma) for active components.
    * **Sparsity Penalty (L1):** Encourages the model to correctly identify the number of components by penalizing non-zero amplitudes for inactive components.
    * **Reconstruction Loss:** Enforces physical consistency by penalizing the difference between the true spectrum and the one reconstructed from the model's predicted parameters.

---

## 🛠️ Technologies Used

* **Python 3.10**
* **Conda** for environment management
* **TensorFlow / Keras** for building and training the neural network
* **Scikit-learn** for data scaling and model evaluation
* **NumPy** & **Pandas** for data manipulation
* **H5Py / PyTables** for efficient data storage
* **Matplotlib / Seaborn** for data visualization

---

## 📂 Project Structure

The repository is organized as follows:

```
.
├── keras_N_component_model/    # Output directory for the trained model & scalers.
│                               # 'N' depends on MAX_COMPONENTS_IN_DATA in config.py.
├── config.py                   # Central configuration file for all parameters.
├── generate_synthetic_spectra.py # Core functions to create single synthetic spectra.
├── generate_spectra_for_ML.py  # Script to generate large HDF5 datasets for training/validation.
├── ML_utils.py                 # Helper functions for plotting, custom losses, and data loading.
├── train_NN_model.py           # Main script to train the primary CNN model (recommended).
├── train_ML_model.py           # Alternative script using traditional ML (e.g., RandomForest).
├── evaluate_model_results.py   # Script to load and evaluate the trained CNN model.
├── .gitignore                  # Specifies files for Git to ignore.
└── README.md                   # You are here!
```

---

## ⚙️ Setup and Installation

To run this project, you'll need to set up the specified Conda environment.

1.  **Clone the repository (once it's on GitHub):**
    ```bash
    git clone https://github.com/ashleyrbemis/spectral-line-fitting-ml.git
    cd spectral-line-fitting-ml
    ```

2.  **Create and activate the Conda environment:**
    The environment is named `spectral_ML_env`.
    ```bash
    # Create the environment with Python 3.10
    conda create -n spectral_ML_env python=3.10

    # Activate the environment
    conda activate spectral_ML_env
    ```

3.  **Install the required libraries:**
    With the environment active, install packages using the following two commands. It's important to install `pytables` with Conda first.
    ```bash
    # First, install PyTables using Conda
    conda install pytables

    # Then, install the remaining packages with Pip
    pip install numpy pandas tensorflow scikit-learn matplotlib seaborn h5py tqdm scipy
    ```

---

## ⚙️ Configuration & Customization

All key parameters for data generation, model training, and evaluation are centralized in `config.py`. By editing this file, you can customize the entire workflow to match the characteristics of different datasets or to experiment with model architectures.

### Data Generation Parameters

* **Dataset Size & Components:**
    * `NUM_TRAINING_SPECTRA`: The number of spectra in the training set.
    * `NUM_VALIDATION_SPECTRA`: The number of spectra in the validation set.
    * `MIN_COMPONENTS_PER_SPECTRUM`: The minimum number of Gaussian components a spectrum can have.
    * `MAX_COMPONENTS_IN_DATA`: The maximum number of Gaussian components a single spectrum can have.

* **Spectral & Component Properties:**
    * `NUM_CHANNELS`: The number of data points (channels) in each 1D spectrum.
    * `RMS_NOISE_RANGE`: The `(min, max)` range for the RMS noise level added to each spectrum.
    * `COMP_AMP_RANGE`: The `(min, max)` range for component amplitude.
    * `COMP_MEAN_RANGE`: The `(min, max)` range for component mean (position).
    * `COMP_SIGMA_RANGE`: The `(min, max)` range for component sigma (width).

### Model & Training Parameters

* **Training Hyperparameters:**
    * `EPOCHS`: The maximum number of training epochs.
    * `BATCH_SIZE`: The number of spectra per batch during training.
    * `LEARNING_RATE`: The initial learning rate for the Adam optimizer.
    * `EARLY_STOPPING_PATIENCE`: Number of epochs with no improvement after which training will be stopped.
    * `EARLY_STOPPING_TOLERANCE`: Minimum change in the monitored quantity to qualify as an improvement.

* **Custom Loss Function Weights:**
    * `SPARSITY_WEIGHT`: Controls the strength of the L1 penalty on inactive component amplitudes.
    * `RECONSTRUCTION_WEIGHT`: Controls the strength of the penalty on the reconstructed spectrum's accuracy.

### Evaluation & Plotting Parameters

* **Plotting Settings:**
    * `NUM_PLOT_SAMPLE_SPECTRA`: The number of sample spectra plotted after the *data generation* script runs.
    * `NUM_PLOTS_EXAMPLES`: The number of sample spectra plotted by the *evaluation* script.
    * `PLOT_AMP_THRESHOLD` / `PLOT_SIGMA_THRESHOLD`: Minimum predicted values for a component to be considered "active" and included in plots.

* **Component Matching Criteria:** (For calculating R² score)
    * `MATCH_MEAN_DIFF_MAX`: Maximum allowed difference in mean for a predicted component to match a true one.
    * `MATCH_SIGMA_DIFF_FACTOR_MAX`: Maximum fractional difference in sigma for a match.
    * `MATCH_AMP_DIFF_FACTOR_MAX`: Maximum fractional difference in amplitude for a match.

---

## 🚀 How to Run the Project

Follow these steps in order from the root directory of the project.

**Step 1: Generate the Training & Validation Data**
This script creates the `spectra_for_ml_training.h5` and `spectra_for_ml_validation.h5` files needed for the next step.
```bash
python generate_spectra_for_ML.py
```

**Step 2: Train the Neural Network Model**
This script will load the training data, build the CNN model, train it, and save the final model and data scalers into the `keras_N_component_model/` directory.
```bash
python train_NN_model.py
```

**Step 3: Evaluate the Trained Model**
This script loads the model you just trained and runs it on the validation dataset to produce performance metrics and visualizations of sample fits.
```bash
python evaluate_model_results.py
```

---

## 📊 Results

The model was evaluated on a hold-out validation set containing a mix of 1- and 2-component spectra. The evaluation focused on two key areas: the accuracy of the predicted Gaussian parameters (a regression task) and the model's ability to identify the correct number of components (a classification task).

### Parameter Prediction Performance

The most meaningful metric for regression is the R² score calculated on only the **active (non-zero) true components**, as this reflects the model's true predictive power on real spectral features.

* **R² Score (Active Parameters):** **0.9235**
    * *Interpretation: The model successfully explains over 92% of the variance in the true parameters of the actual Gaussian components. This indicates a very high level of accuracy and predictive power for the features it identifies.*

#### Performance Breakdown by Parameter Type (Active Components)

The model shows strong performance in predicting the position (Mean) and width (Sigma) of the spectral lines.

| Parameter         | R² Score |
| :---------------- | :------- |
| **Mean (Position)** | **0.7303** |
| **Sigma (Width)** | **0.6434** |
| Amplitude         | 0.2341   |

### Component Count Detection Performance

The model's ability to implicitly detect the number of components was evaluated by counting how many predicted amplitudes were above a physically-motivated threshold.

* **Component Count Accuracy:** **52.7%**
    * *Interpretation: The model correctly identifies the number of components in over half of the validation cases. A confusion matrix is saved to the plots directory for a more detailed breakdown of classification performance.*

---

## ✨ Future Enhancements

Potential improvements and future directions for this project include:

* **Improve Component Counting & Amplitude Prediction:** The model's primary weaknesses are its moderate accuracy in counting components and its lower R² score for amplitude. Future work should focus on architectures or loss functions that specifically target these areas.
* **Implement a Multi-Task Model:** A dedicated classification head could be added to the network to explicitly predict the number of components, which may improve both counting and regression performance.
* **Train on Real Observational Data:** Adapt the pipeline to use real-world astronomical data to test its robustness and practical applicability.
* **Hyperparameter Tuning:** Systematically tune hyperparameters (e.g., learning rate, batch size, network depth) using a library like KerasTuner or Optuna to optimize model performance.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

