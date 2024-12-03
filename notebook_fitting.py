#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats


# In[2]:


# Load data from a TSV file
def load_data_tsv(file_path):

    # Read the sixth row of the file as column names
    columns = pd.read_csv(file_path, nrows=6, header=None, sep='\t', skiprows=5).iloc[0, :].tolist()

    # Read the seventh row of the file as units
    units = pd.read_csv(file_path, nrows=7, header=None, sep='\t', skiprows=6).iloc[0, :].tolist()

    # Read the data section
    data = pd.read_csv(file_path, sep='\t', skiprows=7, names=columns)

    # Return units and data
    return units, data


# In[3]:


# Function to detect the start and end points of rising peaks
def find_rising_peak_start_end(data, signal_column='H2O', gradient_threshold=200):
    
    # Smooth the signal
    data[signal_column + '_smoothed'] = data[signal_column].rolling(window=5).mean()
    
    # Compute the first and second derivatives
    data[signal_column + '_derivative1'] = data[signal_column + '_smoothed'].diff()
    data[signal_column + '_derivative2'] = data[signal_column + '_derivative1'].diff()

    # Initialize lists for peak start and end indices
    peak_starts = []
    peak_ends = []

    # Iterate through the data to find peak start and end points
    in_peak = False
    for i in range(1, len(data) - 1):
        # Check if we are in the rising phase
        if not in_peak:
            # Detect the start of the rising phase
            if data.iloc[i][signal_column + '_derivative1'] > gradient_threshold and \
               data.iloc[i][signal_column + '_derivative2'] < 0:
                peak_starts.append(data.iloc[i]['SECONDS'] + 15)
                in_peak = True
        else:
            # Detect the end of the rising phase
            if data.iloc[i][signal_column + '_derivative1'] < 0 and \
               data.iloc[i][signal_column + '_derivative2'] > 0:
                peak_ends.append(data.iloc[i]['SECONDS'] - 30)
                in_peak = False

    # If the last peak has not ended, use the last data point's SECONDS as the end time
    if in_peak:
        peak_ends.append(data.iloc[-1]['SECONDS'])

    # Create a DataFrame with all the start and end points
    peak_pairs = pd.DataFrame({
        'Start': peak_starts,
        'End': peak_ends
    })

    # Filter out peak pairs where the time difference between start and end is less than 80 seconds
    peak_pairs = peak_pairs[peak_pairs['End'] - peak_pairs['Start'] >= 25]

    return peak_pairs['Start'], peak_pairs['End']


# In[4]:


# Function for linear regression fitting
def linear_regression_fit(data, peak_starts, peak_ends, signal_column):
    # Ensure the signal column is of numeric type
    data[signal_column] = pd.to_numeric(data[signal_column])

    # Extract data points for each peak range
    fits = []
    for i in range(len(peak_starts)):
        df_fit = data[(data['SECONDS'] >= peak_starts.iloc[i]) & (data['SECONDS'] < peak_ends.iloc[i])].copy()  # Copy data to avoid modifying the original dataframe

        # Perform linear regression fitting
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_fit['SECONDS'], df_fit[signal_column])
        except Exception as e:
            print(f"Error fitting data between peaks {peak_starts.iloc[i]} and {peak_ends.iloc[i]}: {e}")
            continue  # Skip the current iteration if an error occurs during fitting

        # Save the fitting results
        fits.append({
            'Start': peak_starts.iloc[i],
            'End': peak_ends.iloc[i],
            'Slope': slope,
            'Intercept': intercept,
            'R-squared': r_value**2,
            'P-value': p_value,
            'Standard Error': std_err
        })
    
    # Create a DataFrame with all the fitting results
    fits_df = pd.DataFrame(fits)
    return fits_df


# In[5]:


# Function to plot raw data and the start/end points of peaks
def plot_data(data, file_name = "haha"):
    
    # Ensure the SECONDS column is of integer type
    data['SECONDS'] = data['SECONDS'].astype(int)
    
    # Create a new figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # Plot H2O concentration over time
    axs[0].plot(data['SECONDS'], data['H2O'], label='H2O (ppm)', color='blue')
    axs[0].set_title('H2O Concentration over Time')
    axs[0].set_xlabel('SECONDS')
    axs[0].set_ylabel('H2O Concentration (ppm)')
    axs[0].legend()

    # Plot CO2 concentration over time
    axs[1].plot(data['SECONDS'], data['CO2'], label='CO2 (ppm)', color='blue')
    axs[1].set_title('CO2 Concentration over Time')
    axs[1].set_xlabel('SECONDS')
    axs[1].set_ylabel('CO2 Concentration (ppm)')
    axs[1].legend()

    # Plot CH4 concentration over time
    axs[2].plot(data['SECONDS'], data['CH4'], label='CH4 (ppb)', color='blue')
    axs[2].set_title('CH4 Concentration over Time')
    axs[2].set_xlabel('SECONDS')
    axs[2].set_ylabel('CH4 Concentration (ppb)')
    axs[2].legend()    

    # Adjust the spacing between subplots
    plt.tight_layout()

    first_peak_start = str(peak_starts.iloc[0])

    # Save the figure as a PNG file
    fig.savefig(file_name+"_raw.png")

    # Display the figure
    plt.show()
    return


# In[6]:


# Function to plot raw data and linear regression results for peaks
def plot_data_subplots(data, gradient_threshold=100, file_name='haha'):
    
    # Find the start and end points of the rising peaks
    peak_starts, peak_ends = find_rising_peak_start_end(data, 'H2O', gradient_threshold)

    # Extract a cycle of data around the peaks
    cycle = data[(data['SECONDS'] >= (peak_starts.iloc[0] - 45)) & (data['SECONDS'] <= (peak_ends.iloc[-1] + 45))].copy()

    # Ensure the SECONDS column is of integer type
    data['SECONDS'] = cycle['SECONDS'].astype(int)
    
    # Create a new figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # Plot H2O concentration over time
    axs[0].plot(data['SECONDS'], data['H2O'], label='H2O (ppm)', color='blue')
    axs[0].set_title('H2O Concentration over Time')
    axs[0].set_xlabel('SECONDS')
    axs[0].set_ylabel('H2O Concentration (ppm)')
    axs[0].legend()

    # Plot CO2 concentration over time
    axs[1].plot(data['SECONDS'], data['CO2'], label='CO2 (ppm)', color='blue')
    axs[1].set_title('CO2 Concentration over Time')
    axs[1].set_xlabel('SECONDS')
    axs[1].set_ylabel('CO2 Concentration (ppm)')
    axs[1].legend()

    # Plot CH4 concentration over time
    axs[2].plot(data['SECONDS'], data['CH4'], label='CH4 (ppb)', color='blue')
    axs[2].set_title('CH4 Concentration over Time')
    axs[2].set_xlabel('SECONDS')
    axs[2].set_ylabel('CH4 Concentration (ppb)')
    axs[2].legend()
    
    # Mark the start and end points of the peaks
    axs[0].scatter(peak_starts, data.loc[peak_starts.index, 'H2O'], color='#FFB000', label='Peak Start', zorder=7)
    axs[0].scatter(peak_ends, data.loc[peak_ends.index, 'H2O'], color='#FFB000', label='Peak End', zorder=7)
    axs[0].legend()

    # Plot linear regression fit for CO2 concentration
    fits_df = linear_regression_fit(data, peak_starts, peak_ends, 'CO2')
    for i in range(len(fits_df)):
        start = fits_df['Start'].iloc[i]
        end = fits_df['End'].iloc[i]
        slope = fits_df['Slope'].iloc[i]
        intercept = fits_df['Intercept'].iloc[i]
        
        # Plot the fitted line
        x_fit = pd.Series(np.linspace(start, end, 100))
        y_fit = slope * x_fit + intercept
        axs[1].plot(x_fit, y_fit, label=f"Fit between {start} and {end}", linestyle='-', color='#FFB000', linewidth=3)

    # Plot linear regression fit for CH4 concentration
    fits_df = linear_regression_fit(data, peak_starts, peak_ends, 'CH4')
    for i in range(len(fits_df)):
        start = fits_df['Start'].iloc[i]
        end = fits_df['End'].iloc[i]
        slope = fits_df['Slope'].iloc[i]
        intercept = fits_df['Intercept'].iloc[i]
        
        # Plot the fitted line
        x_fit = pd.Series(np.linspace(start, end, 100))
        y_fit = slope * x_fit + intercept
        axs[2].plot(x_fit, y_fit, label=f"Fit between {start} and {end}", linestyle='-'


# In[ ]:


# Set the path to the data folder
data_folder_path = os.path.join(os.getcwd(), 'example_fitting')

# Define a function to save a DataFrame to a TSV file
def save_to_tsv(df, filename):
    df.to_csv(filename, sep='\t', index=False)

# Iterate over files in the data folder
for filename in os.listdir(data_folder_path):
    if filename.endswith('.txt'):  # Process only .txt files
        file_path = os.path.join(data_folder_path, filename)
        
        # Load the data
        units, data = load_data_tsv(file_path)
        
        # Find the start and end points of rising peaks
        peak_starts, peak_ends = find_rising_peak_start_end(data, 'H2O', 100)
        print(peak_starts, peak_ends)

        # Perform linear regression fitting for CO2 and CH4
        fits_co2 = linear_regression_fit(data, peak_starts, peak_ends, 'CO2')
        fits_ch4 = linear_regression_fit(data, peak_starts, peak_ends, 'CH4')
        
        # Define filenames for saving the fit results
        co2_filename = f"{file_path}_co2_fit.tsv"
        ch4_filename = f"{file_path}_ch4_fit.tsv"

        # Save the CO2 fit results
        save_to_tsv(fits_co2, co2_filename)
        # Save the CH4 fit results
        save_to_tsv(fits_ch4, ch4_filename)
        

        # Plot raw data
        plot_data(data, file_path)
        
        # Plot raw data and linear regression results
        plot_data_subplots(data, 100, file_path)

