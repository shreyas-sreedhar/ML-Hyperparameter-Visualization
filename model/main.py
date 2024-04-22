import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='ML Hyperparameter Optimization',
    layout='wide')

#---------------------------------#
st.write("""
# Exploring Regression Model Optimization: ML Hyperparameter Visualization

This app allows you to experiment with various regression algorithms and hyperparameters to optimize your machine learning model's performance, visually.
         
Upload your dataset from the sidebar to get started or use one of the dummy datasets available. 
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload Your Dataset')

# Sidebar - Specify parameter settings
uploaded_file = st.sidebar.file_uploader("Upload your CSV file from the sidebar" , type=["csv"])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileSize":uploaded_file.size}
    st.sidebar.write(file_details)

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/shreyas-sreedhar/ML-Hyperparameter-Visualization/main/data/data_1.csv)
""")
#add later

#---------------------------------#
# Sidebar - Specify parameter settings
st.sidebar.header('Select Model and Set Parameters')
# Sidebar - Specify X and Y columns


regression_model = st.sidebar.selectbox('Choose Regression Model', ['Linear Regression', 'Ridge Regression', 'Random Forest Regression'])

split_size = st.sidebar.slider('Data Split Ratio (% for Training Set)', 10, 90, 80, 5)

st.sidebar.subheader('Learning Parameters')

parameter_n_estimators = st.sidebar.slider('Number of Estimators (Random Forest)', 0, 500, (10,50), 50)
parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)

parameter_alpha = st.sidebar.slider('Regularization Strength (Ridge Regression)', 0.01, 10.0, 1.0, 0.01)

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('Dataset')


#---------------------------------#
# Model building

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href

# Update the function to include 3D plotting
def build_model(df, regression_model):
    # Check if dataset contains non-numeric values
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    if not numerical_columns:
        st.error("Insufficient numerical values to perform regression.")
        return
    default_index = len(numerical_columns) - 1
    selected_Y_column = st.selectbox('Select Y Column', numerical_columns, index=default_index)
    X = df[numerical_columns]
    Y = df[selected_Y_column] # Selecting the last numerical column as Y
    st.markdown(f'A model is being built to predict the following **{Y.name}** variable using {regression_model}.')
    if Y.isnull().any():
        st.error("Target variable contains NaN values. Please handle missing values before proceeding.")
        return

        
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    if regression_model == 'Linear Regression':
        model = LinearRegression()
        param_grid = {}
        hyperparams = []
    elif regression_model == 'Ridge Regression':
        model = Ridge()
        param_grid = {'alpha': [parameter_alpha]}
        hyperparams = ['alpha']
    elif regression_model == 'Random Forest Regression':
        model = RandomForestRegressor()
        n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
        max_features_range = ['auto', 'sqrt', 'log2']  # You can modify this according to your needs
        param_grid = {'n_estimators': n_estimators_range, 'max_features': max_features_range}
        hyperparams = ['max_features', 'n_estimators']

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid.fit(X_train, Y_train)

    st.subheader('Model Performance')

    Y_pred_test = grid.predict(X_test)
    st.write('Coefficient of Determination ($R^2$):')
    st.info(r2_score(Y_test, Y_pred_test))

    st.write('Error (Mean Squared Error or Mean Absolute Error):')
    st.info(mean_squared_error(Y_test, Y_pred_test))

    st.write("The best parameters are %s with a score of %0.2f"
                % (grid.best_params_, grid.best_score_))

    st.subheader('Model Parameters')
    st.write(grid.get_params())

    # -----Process grid data-----#
    if hyperparams:
        grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]), pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])], axis=1)
        # Segment data into groups based on the hyperparameters
        grid_contour = grid_results.groupby(hyperparams).mean()
        # Pivoting the data
        grid_reset = grid_contour.reset_index()
        grid_reset.columns = hyperparams + ['R2']
        if len(hyperparams) == 2:
            grid_pivot = grid_reset.pivot_table(values='R2', index=hyperparams[0], columns=hyperparams[1])
            x = grid_pivot.columns.values
            y = grid_pivot.index.values
            z = grid_pivot.values

            # -----Plot-----#
            layout = go.Layout(
                scene=dict(
                    xaxis_title=hyperparams[1],
                    yaxis_title=hyperparams[0],
                    zaxis_title='R2',
                    xaxis=dict(tickmode='linear'),
                    yaxis=dict(tickmode='linear')
                )
            )
            fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
            fig.update_layout(title='Hyperparameter Tuning',
                                autosize=False,
                                width=800,
                                height=800,
                                margin=dict(l=65, r=50, b=65, t=90))
            st.plotly_chart(fig)
        else:
            st.write("Only one hyperparameter to visualize for this model.")
            if regression_model == 'Ridge Regression':
                model = Ridge()
                alphas = np.logspace(-3, 2, 50)  # Range of alpha values to test
                scores = []
                for alpha in alphas:
                    ridge = Ridge(alpha=alpha)
                    ridge.fit(X_train, Y_train)
                    score = ridge.score(X_test, Y_test)
                    scores.append(score)

                st.subheader('Ridge Regression Hyperparameter Tuning')

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=alphas, y=scores, mode='lines', name='R^2 Score'))
                fig.update_layout(
                    xaxis_title='Alpha',
                    yaxis_title='R^2 Score',
                    xaxis_type='log',
                    title='Ridge Regression Hyperparameter Tuning'
                )

                st.plotly_chart(fig)
    else:
        st.write("No hyperparameters to visualize for this model.")
#---------------------------------#
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    build_model(df, regression_model)
else:
    st.info('Please upload a CSV file or use the example dataset.')
    if st.button('Use Example Dataset'):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='Response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('**Diabetes dataset** is used as the example dataset.')
        st.write(df.head(5))

        build_model(df, regression_model)
