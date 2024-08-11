import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True, dayfirst=True)
    return data

# Remove outliers
def remove_outliers(df):
    q1 = df['Close'].quantile(0.25)
    q3 = df['Close'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df['Close'] >= lower_bound) & (df['Close'] <= upper_bound)]
    return df

# Standardize data
def standardize_data(train, test):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[['Close']])
    test_scaled = scaler.transform(test[['Close']])
    train['Close'] = train_scaled
    test['Close'] = test_scaled
    return train, test, scaler

# KPSS Test
def kpss_test(timeseries):
    result = kpss(timeseries, regression='c', nlags='auto')
    return result

# ADF Test
def adf_test(timeseries):
    result = adfuller(timeseries)
    return result

# SARIMAX Model
def sarimax_model(train, order, seasonal_order):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    results = model.fit()
    return results

# Forecasting
def forecast(model, steps):
    pred = model.get_forecast(steps=steps)
    return pred

# Plotting
def plot_forecast(actual, predicted):
    fig, ax = plt.subplots(figsize=(12, 6))
    actual.index = pd.to_datetime(actual.index)
    predicted.index = pd.to_datetime(predicted.index)
    sns.lineplot(data=actual, label='Actual', ax=ax)
    sns.lineplot(data=predicted, label='Predicted', ax=ax)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Prices')
    plt.legend()
    st.pyplot(fig)

# Main Streamlit App
def main():
    st.title("Stock Price Prediction using SARIMAX")
    
    # Upload Data
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Original Data")
        st.write(data)

        # Remove Outliers
        data = remove_outliers(data)
        st.write("Data after removing outliers")
        st.write(data)
        
        # Split data into train and test
        train = data.iloc[:-30]
        test = data.iloc[-30:]
        
        # Standardize Data
        train, test, scaler = standardize_data(train, test)
        st.write("Standardized Train Data")
        st.write(train)
        st.write("Standardized Test Data")
        st.write(test)
        
        # KPSS Test
        kpss_result = kpss_test(train['Close'])
        st.write("KPSS Test")
        st.write(f"KPSS Statistic: {kpss_result[0]}")
        st.write(f"p-value: {kpss_result[1]}")
        st.write(f"Lags Used: {kpss_result[2]}")
        st.write("Critical Values:")
        st.write(kpss_result[3])
        
        # ADF Test
        adf_result = adf_test(train['Close'])
        st.write("ADF Test")
        st.write(f"ADF Statistic: {adf_result[0]}")
        st.write(f"p-value: {adf_result[1]}")
        st.write(f"Lags Used: {adf_result[2]}")
        st.write("Critical Values:")
        st.write(adf_result[4])
        
        # Model Training
        model = sarimax_model(train['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        
        # Forecast
        forecast_steps = 30  # Forecasting next 30 days
        pred = forecast(model, steps=forecast_steps)
        
        # Inverse transform predictions
        predicted = scaler.inverse_transform(pred.predicted_mean.values.reshape(-1, 1))
        predicted = pd.Series(predicted.flatten(), index=pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D'))
        
        # Print Forecasted Data
        st.write("Next 30 Days Forecasted Close Prices:")
        st.write(predicted)

        # Plotting
        plot_forecast(data['Close'], predicted)

if __name__ == "__main__":
    main()
