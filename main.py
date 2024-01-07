# Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import flask
from flask import Flask, request, jsonify, render_template
import pickle
import requests
import json 

# API KEY for Alpha Vantage 
API_KEY = 'RKZ7ORSHISUE389M'

# Function to get stock data from Alpha Vantage
def get_stock_data(functions, symbol):
    """
    Function to get stock data from Alpha Vantage
    
    Parameters
    ----------
    functions : str
        The time series of your choice. In this case, we want the daily time series.
    symbol : str
        The name of the stock you want to get data for.
        
    Returns
    -------
    df : pandas.DataFrame
        A pandas DataFrame with the stock data.
        
    """
    url = f"https://www.alphavantage.co/query?function={functions}&symbol={symbol}&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df = df.reset_index()
    df = df.rename(columns={'index': 'date', '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    return df

# Print the stock data
def print_stock_data(df):
    """
    Function to print stock data
    
    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame with the stock data.
        
    Returns
    -------
    df : pandas.DataFrame
        A pandas DataFrame with the stock data.
        
    """
    print(df.head())
    print(df.tail())
    print(df.info())
    print(df.describe())
    print(df.columns)
    
    return df

# Function to plot stock data
def plot_stock_data(df):
    """
    Function to plot stock data
    
    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame with the stock data.
        
    Returns
    -------
    df : pandas.DataFrame
        A pandas DataFrame with the stock data.
        
    """
    plt.figure(figsize=(12, 8))
    plt.plot(df['date'], df['close'], label='Close')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Over Time')
    plt.legend()
    plt.show()
    
    return df

# Call in all the functions
def main():
    """
    Function to call in all the functions
    
    Parameters
    ----------
    None
        
    Returns
    -------
    None
        
    """
    # Get the stock data
    df = get_stock_data('TIME_SERIES_DAILY', 'AAPL')
    # Print the stock data
    df = print_stock_data(df)
    # Plot the stock data
    df = plot_stock_data(df)
    
    return None

# Call in the main function
if __name__ == '__main__':
    main()