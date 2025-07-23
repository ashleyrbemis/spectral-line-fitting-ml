# Machine Learning for Spectral Line Fitting

## 📖 Overview

This project implements a deep learning model (a Convolutional Neural Network) to analyze spectral data and automatically identify the number and parameters of underlying Gaussian components. Traditional fitting methods can be slow and require good initial guesses, whereas this ML approach aims to provide a fast and accurate initial analysis of complex spectra.

**Problem Statement:** Given a noisy one-dimensional spectrum, the goal is to predict the parameters (Amplitude, Mean, Sigma) for each Gaussian component present.

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

