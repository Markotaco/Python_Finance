#!/usr/bin/env python
# coding: utf-8

# # Generating the Sharpe Ratio of a stock portfolio

# The intended purpose for this file is to take in a dictionary of stocks and the amount invested in each stocks and to generate a Sharpe Ratio of the portfolio. It is meant to be an easy tool for those looking to evaluate their stock portfolios. Due to the simplicity of the tool, this is also **NOT** meant for day traders, but instead people who are taking a more long-term perspective to investing.

# ---

# ## Necessary Libraries

# In[23]:


import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt


# ## Importing data from stock_sheet

# First, the program will read in the stock names and amount invested from the excel sheet as a pandas dataframe. After which, the ticker names will be stored as an array for use in the downloading of the stock data

# In[24]:


# Reading in stock data from pandas dataframe
stock_details_excel = pd.read_excel("stock_input_file.xlsx",
                                    sheet_name="stock_sheet").sort_values(by="Stock Ticker")
stock_details_excel


# In[25]:


# Extracting the names as an array and storing it as a variable
stock_names_array = np.array(stock_details_excel['Stock Ticker'])
stock_names_array


# ## Transforming the investment values into portfolio weights

# The user would input the amount they have invested into each asset, but what the program needs is the weight of the asset with regards to the whole portfolio

# In[26]:


# Extracting the values from the datadrame, turning it into an array, before dividing by the sum of the values to get the weights
stock_weights_array = np.array(stock_details_excel['Amount Invested'])/sum(stock_details_excel['Amount Invested'])
stock_weights_array


# ## Obtaining the other relevant parameter from the other_parameters sheet

# The program will also need the other relevant data inputted by the user, such as the annual risk-free rate, the time interval the user is looking at and the start and end date of the data the user would like to base their returns off

# In[27]:


other_parameters = pd.read_excel("stock_input_file.xlsx",
                                 sheet_name="other_parameters")
other_parameters


# In[28]:


# Extracting the risk-free rate
risk_free_rate = other_parameters.iloc[0][0]
# Extracting the time interval
return_type = other_parameters.iloc[0][1]
# Extracting the start date and turning it into a string
start_date = str(other_parameters.iloc[0][2].date())
# Extracting the end date and turning it into a string
end_date = str(other_parameters.iloc[0][3].date())

# Viewing Variables
print("Other parameters have been set as the following:")
print(f"Risk-Free Rate: {round(risk_free_rate*100, 2)}%")
print(f"Time Interval of Asset Price: {return_type}")
print(f"Starting Date of Data: {start_date}")
print(f"Ending Date of Data: {end_date}")


# ## Getting the data

# The data will be taken from yfinance, an open source API for historical stock price data. The closing price will be used since that reflects the state of the stocks once the variations during the day has passed over. The return of the stock is simply the percentage change in the stock's closing price from the previous period.

# In[29]:


# Using yf.download to get the data of multiple stocks' closing prices. After that using the pct_change method to get the percentage change in the stock's price from the previous period.
stock_returns = yf.download(tuple(stock_names_array), start=start_date, end=end_date, interval=return_type)['Close'].pct_change()

stock_returns


# ## Calculating the portfolio's expected returns

# A portfolio of size $n$ has a expected weighted return of:
# 
# $$E[Return]$$ 
# $$= E[w_1*r_1 + w_2*r_2 + ... + w_n*r_n]$$
# $$  = w_1*E[r_1] + w_2*E[r_2] + ... + w_n*E[r_n]$$
# 
# In a sample, this translates to $\bar{X_i}$ instead of $E[r_i]$

# In[30]:


# Getting the sample mean of each stock
exp_return = stock_returns.mean()
# Turning it into an array
exp_return_array = np.array(exp_return)


# In[31]:


# Getting the weighted expected returns by multiplying the weights to the sample means
weighted_exp_return = exp_return_array * stock_weights_array
# Getting the sum for the portfolio's expected returns
portfolio_expected_return = weighted_exp_return.sum()
portfolio_expected_return


# ## Calculating the portfolio's variance and standard deviation (volatility)

# A portfolio's return variance can be given by the following equation:
# 
# $$var(Return)$$
# $$= var(w_1*r_1 + w_2*r_2 + ... + w_n*r_n)$$
# $$= w_1^2*var(r_1) + w_2^2*var(r_2) + ... + w_n^2*var(r_n) + 2*w_1*w_2*cov(r_1,r_2) + 2*w_1*w_3*cov(r_1,r_3) + ... 2*w_n*w_{n-1}*cov(w_n,w_{n-1})$$
# 
# In a sample, this would be replaced by sample variances and covariances. The right degrees of freedom are already considered by the method by default

# In[32]:


# Getting the covariance matrix from the stock data
stock_returns_cov = stock_returns.cov()
stock_returns_cov


# In[33]:


# Generating the weights matrix, which consists of a matrix mulitplication between the weights array and its transpose
stock_weights_matrix = np.outer(stock_weights_array, stock_weights_array)
stock_weights_matrix


# The scalar multiplication of the above matrices, followed by a summation accross all elements of the resulting matrix, will give one the portfolio variance as described in the equation above.

# In[34]:


# Multiplying the two matrices, and then getting the sum over all rows, then columns
portfolio_variance = (stock_returns_cov * stock_weights_matrix).sum().sum()
portfolio_variance


# After obtaining the variance of a portfolio, the standard deviation can be given by:
# $$\sigma = \sqrt{var(Return)}$$

# In[35]:


# Obtaining the standard deviation of the portfolio
portfolio_sd = np.sqrt(portfolio_variance)
portfolio_sd


# ## Calculating the Sharpe ratio

# The Sharpe ratio can be calculated as the following:
# 
# $${Sharpe Ratio} = \frac{r_p - r_f}{\sigma_p}$$
# 
# Where $r_p$ and $\sigma_p$ are the expected return of the portfolio and the volatility of the portfolio respectively, and $r_f$ is the return from the risk-free option

# To obtain an appropriate risk-free rate that has been adjusted for the time period the user specified to look at, the annual risk-free rate given by the user needs to be adjusted to fit their time period. Thus, a function will help to do this

# In[36]:


# Creating a function to adjust the annual risk-free rate according to the time interval specified by the user. This ensures a consistent risk-free rate used
def rf_adjuster(rf_input, time_interval):
    "Takes the inputted risk-free rate and returns the interest rate adjusted for the inputted time interval the user is looking at"
    # Looking at the case of daily returns
    if 'd' in time_interval:
        # Converting the annual rate into a daily rate
        rf_base = (1+rf_input)**(1/365) - 1
        # Converting the daily rate to the appropriate period
        rf_adjusted = (1+rf_base)**(int(time_interval[0])) - 1
    # Looking at the case of weekly returns
    elif 'wk' in time_interval:
        # Converting the annual rate into a weekly rate
        rf_base = (1+rf_input)**(1/52) - 1
        # Converting the weekly rate to the appropriate period
        rf_adjusted = (1+rf_base)**(int(time_interval[0])) - 1
    # Looking at the case of monthly returns
    elif 'mo' in time_interval:
        # Converting the annual rate into a monthly rate
        rf_base = (1+rf_input)**(1/12) - 1
        # Converting the monthly rate to the appropriate period
        rf_adjusted = (1+rf_base)**(int(time_interval[0])) - 1
    # If none of the cases match, then the user has given an erroneous input
    else:
        return print("Error, interval given is not a valid interval within this tool")
    # Return the adjusted input
    return rf_adjusted


# In[37]:


# Getting the adjusted risk-free rate
adjusted_risk_free_rate = rf_adjuster(risk_free_rate, return_type)
adjusted_risk_free_rate


# In[38]:


# Getting the Sharpe ratio as defined above
sharpe_ratio = (portfolio_expected_return - adjusted_risk_free_rate)/portfolio_sd
sharpe_ratio


# ## Portfolio's Final Metrics

# In[39]:


print(f"The Portfolio's Sharpe Ratio is {round(sharpe_ratio, 4)}")


# In[40]:


print(f"The Portfolio's Expected Return for the time interval of '{return_type}' is {round(portfolio_expected_return*100, 2)}%")


# In[41]:


print(f"The Portfolio's Volatility for the time interval of '{return_type}' is {round(portfolio_sd*100, 2)}%")


# ## Writing the metrics into an excel output file

# These metrics can also be written into output excel file for easier viewing. This can be done with pandas, but that means that the data first needs to be wrapped up in a dataframe

# In[42]:


# Storing the data to show in lists
metric_name = ['Sharpe Ratio', 'Portfolio Expected Return', 'Portfolio Volatility']
metric_value = [round(sharpe_ratio, 4), round(portfolio_expected_return, 4), round(portfolio_sd, 4)]

# Creating the dataframe
metric_table = pd.DataFrame({'Metric Name': metric_name,
                             'Value': metric_value})
metric_table


# It would also be useful to write in a separate sheet the input details of what the user gave to the programme, so that it would be easier for the user to keep track of their excel outputs. For this the pandas dataframes already exist, which are the original input tables `stock_details_excel` and `other_paramters`. However, `other_parameters`'s datetimes need to be first turned into dates only

# In[43]:


# Getting today's date and current time at which the file is generated
date_today = dt.datetime.today().strftime('%Y-%m-%d--%H%M%S')
# Transforming the start and end datetimes into dates only
other_parameters['Start Date'][0] = other_parameters['Start Date'][0].date()
other_parameters['End Date'][0] = other_parameters['End Date'][0].date()
# Using the excel writer object
with pd.ExcelWriter(f"Excel_Outputs/Output-{date_today}.xlsx") as writer:
    # Writing the metrics into a excel sheet
    metric_table.to_excel(writer,
                         sheet_name='Metrics',
                         index=False)
    # Adding a sheet of the inputted stocks into the written excel sheet
    stock_details_excel.to_excel(writer,
                         sheet_name='Inputted Stocks',
                         index=False)
    # Adding a sheet of the other parameters given by the useer
    other_parameters.to_excel(writer,
                         sheet_name='Inputted Misc.',
                         index=False)


# In[ ]:




