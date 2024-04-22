# ML Hyperparameter Optimization App

Welcome to the ML Hyperparameter Optimization App! This Streamlit web application allows you to experiment with various regression algorithms and hyperparameters to optimize your machine learning model's performance.

## Features

- **Regression Algorithms:** Choose between Linear Regression, Ridge Regression, and Random Forest Regression.
- **Hyperparameter Tuning:** Experiment with different hyperparameters such as regularization strength and number of estimators to optimize model performance.
- **Data Visualization:** Visualize the hyperparameter tuning process and model performance with interactive plots generated using Plotly.

## Usage

1. **Upload Your Dataset:** Upload your CSV dataset or use the provided example dataset.
2. **Select Model and Set Parameters:** Choose a regression model and set hyperparameters.
3. **Explore Model Performance:** Visualize the optimization process and view model performance metrics such as coefficient of determination (R^2) and error (mean squared error or mean absolute error).

## Installation

To run the app locally, follow these steps:

1. Clone the repository:
   ``git clone <repository-url> && cd <repository-name>``
2. Install the required dependencies:
   ``pip install -r requirements.txt``
3. Run the app:
    ``streamlit run app.py``
## Example Dataset

An example CSV dataset is provided for demonstration purposes. You can use this dataset to explore the app's features without uploading your own data.

## Technologies Used

- Python
- Streamlit: For building the interactive web application.
- Pandas: For data manipulation and preprocessing.
- NumPy: For numerical computations.
- Plotly: For creating interactive data visualizations.
- scikit-learn: For implementing regression algorithms and hyperparameter tuning.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

