# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import itertools

# Set the page configuration
st.set_page_config(page_title='Bike Rental Prediction App', page_icon='🚴‍♂️', layout='centered')
# Main title for the app
st.title("Bike Rental Predictor App 🚲")


# Load the trained and tuned pipeline
pipeline = joblib.load('bike_rental_pipeline_tuned.pkl')

# Load the dataset
data = pd.read_csv('bikes_sample.csv')

# Convert 'dteday' to datetime
data['dteday'] = pd.to_datetime(data['dteday'])

# Remove unnecessary columns for EDA
eda_data = data.drop(['instant', 'casual', 'registered'], axis=1)

# Create a sidebar navigation
st.sidebar.title('Navigation')
selection = st.sidebar.radio('Go to', ['Exploratory Data Analysis', 'Bike Rental Prediction'])

if selection == 'Exploratory Data Analysis':
    st.title("Exploratory Data Analysis")

    # Sidebar for user interaction in EDA
    st.sidebar.header('Filter Data for EDA')

    # Filter by season
    season_options = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    selected_season = st.sidebar.multiselect('Season', options=list(season_options.keys()),
                                             format_func=lambda x: season_options[x],
                                             default=list(season_options.keys()))

    # Filter data based on selection
    filtered_data = eda_data[eda_data['season'].isin(selected_season)]

    # Interactive Line Chart of Bike Rentals Over Time using Plotly
    st.subheader('Bike Rentals Over Time')
    fig = px.line(filtered_data, x='dteday', y='cnt', color=filtered_data['season'].map(season_options),
                  labels={'cnt': 'Total Rentals', 'dteday': 'Date', 'color': 'Season'})
    st.plotly_chart(fig, use_container_width=True)

    # Interactive Histogram of Bike Rentals
    st.subheader('Distribution of Bike Rental Counts')
    fig = px.histogram(filtered_data, x='cnt', nbins=30, marginal='box', color_discrete_sequence=['indianred'])
    fig.update_layout(xaxis_title='Total Rentals', yaxis_title='Frequency')
    st.plotly_chart(fig, use_container_width=True)

    # Interactive Scatter Plot
    st.subheader('Temperature vs. Bike Rentals')
    fig = px.scatter(filtered_data, x='temp', y='cnt', color=filtered_data['season'].map(season_options),
                     labels={'temp': 'Temperature (°C)', 'cnt': 'Total Rentals', 'color': 'Season'})
    st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap using Plotly Figure Factory
    st.subheader('Correlation Heatmap')
    corr = filtered_data.corr()
    x = list(corr.columns)
    y = list(corr.index)
    z = np.round(corr.values, 2)

    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis')
    fig.update_layout(width=700, height=700)
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance Visualization
    if st.checkbox('Show Feature Importances'):
        st.subheader('Feature Importances')
        # Extract feature importances from the model
        importances = pipeline.named_steps['model'].feature_importances_
        # Get the names of the one-hot encoded features
        ohe_feature_names = pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(
            input_features=['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit'])
        feature_names = list(ohe_feature_names) + ['temp', 'hum', 'windspeed']

        feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Plot feature importances using Plotly
        fig = px.bar(feature_importances.head(10), x='Importance', y='Feature', orientation='h')
        fig.update_layout(title='Top 10 Feature Importances', yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)

elif selection == 'Bike Rental Prediction':
    st.title('Bike Rental Count Prediction')

    st.write("""
    This app predicts the **Bike Rental Count** based on user input parameters.
    """)

    # Function to get user input
    def user_input_features():
        st.sidebar.header('User Input Parameters')

        season = st.sidebar.selectbox('Season', options=[1, 2, 3, 4],
                                      format_func=lambda x: {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}.get(x))

        # Define months corresponding to seasons
        season_months = {
            1: [3, 4, 5],     # Spring
            2: [6, 7, 8],     # Summer
            3: [9, 10, 11],   # Fall
            4: [12, 1, 2]     # Winter
        }

        # Update month options based on selected season
        months_options = season_months[season]
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                       5: 'May', 6: 'June', 7: 'July', 8: 'August',
                       9: 'September', 10: 'October', 11: 'November', 12: 'December'}

        mnth = st.sidebar.selectbox('Month', options=months_options,
                                    format_func=lambda x: month_names[x])

        holiday = st.sidebar.selectbox('Holiday', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        weekday = st.sidebar.selectbox('Weekday', options=[0, 1, 2, 3, 4, 5, 6],
                                       format_func=lambda x: {
                                           0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
                                           4: 'Thursday', 5: 'Friday', 6: 'Saturday'}.get(x))
        workingday = st.sidebar.selectbox('Working Day', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        weathersit = st.sidebar.selectbox('Weather Situation', options=[1, 2, 3, 4],
                                          format_func=lambda x: {
                                              1: 'Clear, Few clouds',
                                              2: 'Mist + Cloudy',
                                              3: 'Light Snow/Rain',
                                              4: 'Heavy Rain/Ice'}.get(x))
        temp = st.sidebar.slider('Temperature (°C)', min_value=-10.0, max_value=40.0, value=20.0)
        hum = st.sidebar.slider('Humidity (%)', 0, 100, 50)
        windspeed = st.sidebar.slider('Windspeed (km/h)', min_value=0.0, max_value=56.0, value=10.0)

        data = {
            'Season': {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}[season],
            'Month': month_names[mnth],
            'Holiday': 'Yes' if holiday == 1 else 'No',
            'Weekday': {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
                        4: 'Thursday', 5: 'Friday', 6: 'Saturday'}[weekday],
            'Working Day': 'Yes' if workingday == 1 else 'No',
            'Weather Situation': {1: 'Clear, Few clouds',
                                  2: 'Mist + Cloudy',
                                  3: 'Light Snow/Rain',
                                  4: 'Heavy Rain/Ice'}[weathersit],
            'Temperature (°C)': temp,
            'Humidity (%)': hum,
            'Windspeed (km/h)': windspeed
        }

        model_data = {
            'season': season,
            'mnth': mnth,
            'holiday': holiday,
            'weekday': weekday,
            'workingday': workingday,
            'weathersit': weathersit,
            'temp': temp,
            'hum': hum,
            'windspeed': windspeed
        }

        features_display = pd.DataFrame(data, index=[0])
        features = pd.DataFrame(model_data, index=[0])
        return features, features_display

    # Get user input
    input_df, input_display = user_input_features()

    # Prediction button
    if st.button('Predict'):
        # Display user input
        st.header('User Input Parameters')
        # Display the table with correct parameter names
        st.table(input_display.T)

        # Predict using the tuned pipeline
        prediction = pipeline.predict(input_df)
        predicted_count = int(prediction[0])

        st.header('Predicted Bike Rental Count')
        st.write(f"**{predicted_count}** bikes are predicted to be rented.")

    else:
        st.write('Click the **Predict** button to see the prediction.')
