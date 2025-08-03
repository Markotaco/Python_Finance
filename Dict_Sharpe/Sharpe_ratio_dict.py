#!/usr/bin/env python
# coding: utf-8

# # Generating the Sharpe Ratio of a stock portfolio

# The intended purpose for this file is to take in a dictionary of stocks and the amount invested in each stocks and to generate a Sharpe Ratio of the portfolio. It is meant to be an easy tool for those looking to evaluate their stock portfolios. Due to the simplicity of the tool, this is also **NOT** meant for day traders, but instead people who are taking a more long-term perspective to investing.

# ---

# ## Input Fields
# Please fill in the stock tickers and the associated amount invested into the following fields. Ensure that the values are in the same currency. Please also fill in the annual risk-free rate you would like it to be evaluated at.

# In[9]:


stock_details = {'aapl': 300, 'msft': 300, 'nvda': 400, 'gld': 800}
risk_free_rate = 0.03


# Please also fill in the the start and end dates of the historical period you would like to base the expected returns on.

# In[10]:


start_date = '2015-07-01'
end_date = '2025-07-01'


# Lastly, please fill in the type of returns you would like to see. You may choose from the following options:
# - 1 day: `1d`
# - 5 days: `5d`
# - 1 week: `1wk`
# - 1 month: `1mo`
# - 3 months:`3mo`

# In[11]:


return_type = '1mo'


# Once all inputs are set, please press `run all`

# ---

# ## Necessary Libraries

# In[12]:


import numpy as np
import pandas as pd
import yfinance as yf


# ## Arranging the dataframe by the ticker's alphabetical order

# In[13]:


stock_details = dict(sorted(stock_details.items()))
stock_details


# ## Transforming the investment values into portfolio weights

# The user would input the amount they have invested into each asset, but what the programme needs is the weight of the asset with regards to the whole portfolio

# In[14]:


# Extracting the values from the dictionary, turning it into a tuple and then an array, before dividing by the sum of the values to get the weights
stock_weights_array = np.array(tuple(stock_details.values()))/sum((stock_details.values()))
stock_weights_array


# ## Getting the data

# The data will be taken from yfinance, an open source API for historical stock price data. The closing price will be used since that reflects the state of the stocks once the variations during the day has passed over. The return of the stock is simply the percentage change in the stock's closing price from the previous period.

# In[15]:


# Using yf.download to get the data of multiple stocks' closing prices. After that using the pct_change method to get the percentage change in the stock's price from the previous period.
stock_returns = yf.download(tuple(stock_details.keys()), start=start_date, end=end_date, interval=return_type)['Close'].pct_change()

stock_returns


# ## Calculating the portfolio's expected returns

# A portfolio of size $n$ has a expected weighted return of:
# 
# $$E[Return]$$ 
# $$= E[w_1*r_1 + w_2*r_2 + ... + w_n*r_n]$$
# $$  = w_1*E[r_1] + w_2*E[r_2] + ... + w_n*E[r_n]$$
# 
# In a sample, this translates to $\bar{X_i}$ instead of $E[r_i]$

# In[16]:


# Getting the sample mean of each stock
exp_return = stock_returns.mean()
# Turning it into an array
exp_return_array = np.array(exp_return)


# In[17]:


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

# In[18]:


# Getting the covariance matrix from the stock data
stock_returns_cov = stock_returns.cov()
stock_returns_cov


# In[19]:


# Generating the weights matrix, which consists of a matrix mulitplication between the weights array and its transpose
stock_weights_matrix = np.outer(stock_weights_array, stock_weights_array)
stock_weights_matrix


# The scalar multiplication of the above matrices, followed by a summation accross all elements of the resulting matrix, will give one the portfolio variance as described in the equation above.

# In[20]:


# Multiplying the two matrices, and then getting the sum over all rows, then columns
portfolio_variance = (stock_returns_cov * stock_weights_matrix).sum().sum()
portfolio_variance


# After obtaining the variance of a portfolio, the standard deviation can be given by:
# $$\sigma = \sqrt{var(Return)}$$

# In[21]:


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

# In[22]:


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


# In[23]:


# Getting the adjusted risk-free rate
adjusted_risk_free_rate = rf_adjuster(risk_free_rate, return_type)
adjusted_risk_free_rate


# In[24]:


# Getting the Sharpe ratio as defined above
sharpe_ratio = (portfolio_expected_return - adjusted_risk_free_rate)/portfolio_sd
sharpe_ratio


# ## Portfolio's Final Metrics

# In[25]:


print(f"The Portfolio's Sharpe Ratio is {round(sharpe_ratio, 4)}")


# In[26]:


print(f"The Portfolio's Expected Return for the time interval of '{return_type}' is {round(portfolio_expected_return*100, 2)}%")


# In[27]:


print(f"The Portfolio's Volatility for the time interval of '{return_type}' is {round(portfolio_sd*100, 2)}%")


# In[ ]:




