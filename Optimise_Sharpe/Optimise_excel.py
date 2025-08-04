#!/usr/bin/env python
# coding: utf-8

# # Generating the Sharpe Ratio of a stock portfolio

# The intended purpose for this file is to take in a dictionary of stocks and the amount invested in each stocks and to generate a Sharpe Ratio of the portfolio. It is meant to be an easy tool for those looking to evaluate their stock portfolios. Due to the simplicity of the tool, this is also **NOT** meant for day traders, but instead people who are taking a more long-term perspective to investing.

# ---

# ## Necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import scipy.optimize as opt


# ## Importing data from stock_sheet

# First, the program will read in the stock names and amount invested from the excel sheet as a pandas dataframe. After which, the ticker names will be stored as an array for use in the downloading of the stock data

# In[2]:


# Reading in stock data from pandas dataframe, and sorting the tickers by alphabetical orders
stock_details_excel = pd.read_excel("stock_input_file.xlsx",
                                    sheet_name="stock_sheet").sort_values(by="Stock Ticker")
stock_details_excel


# In[3]:


# Extracting the names as an array and storing it as a variable
stock_names_array = np.array(stock_details_excel['Stock Ticker'])
stock_names_array


# ## Transforming the investment values into portfolio weights

# The user would input the amount they have invested into each asset, but what the program needs is the weight of the asset with regards to the whole portfolio

# In[4]:


# Extracting the values from the datadrame, turning it into an array, before dividing by the sum of the values to get the weights
stock_weights_array = np.array(stock_details_excel['Amount Invested'])/sum(stock_details_excel['Amount Invested'])
stock_weights_array


# ## Obtaining the other relevant parameter from the other_parameters sheet

# The program will also need the other relevant data inputted by the user, such as the annual risk-free rate, the time interval the user is looking at and the start and end date of the data the user would like to base their returns off

# In[5]:


other_parameters = pd.read_excel("stock_input_file.xlsx",
                                 sheet_name="other_parameters")
other_parameters


# In[6]:


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

# In[7]:


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

# In[8]:


# Getting the sample mean of each stock
exp_return = stock_returns.mean()
# Turning it into an array
exp_return_array = np.array(exp_return)


# In[9]:


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

# In[10]:


# Getting the covariance matrix from the stock data
stock_returns_cov = stock_returns.cov()
stock_returns_cov


# In[11]:


# Generating the weights matrix, which consists of a matrix mulitplication between the weights array and its transpose
stock_weights_matrix = np.outer(stock_weights_array, stock_weights_array)
stock_weights_matrix


# The scalar multiplication of the above matrices, followed by a summation accross all elements of the resulting matrix, will give one the portfolio variance as described in the equation above.

# In[12]:


# Multiplying the two matrices, and then getting the sum over all rows, then columns
portfolio_variance = (stock_returns_cov * stock_weights_matrix).sum().sum()
portfolio_variance


# After obtaining the variance of a portfolio, the standard deviation can be given by:
# $$\sigma = \sqrt{var(Return)}$$

# In[13]:


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

# In[14]:


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


# In[15]:


# Getting the adjusted risk-free rate
adjusted_risk_free_rate = rf_adjuster(risk_free_rate, return_type)
adjusted_risk_free_rate


# In[16]:


# Getting the Sharpe ratio as defined above
sharpe_ratio = (portfolio_expected_return - adjusted_risk_free_rate)/portfolio_sd
sharpe_ratio


# ## Optimisation of the Portfolio

# The program can be taken one step further, and be used to generate the investments amounts and weights needed to obtain an optimal portfolio. The portfolio will be optimised by maximising the portfolio's Sharpe Ratio

# For the optimisation code to work, an objective function for the Sharpe Ratio needs to be created. This function will consist of two smaller functions:
# 1. A function calculating the expected return of the portfolio
# 2. A function calculating the volatility of the portfolio

# ### Expected Return Function

# In[17]:


# Function takes a dataframe and a weight array
def expected_return_cal(df, weight):
    # Individual stock's expected return
    indiv_return = np.array(df.mean())
    # Portfolio expected return
    port_return = (indiv_return*weight).sum()
    # Returning the expected return of the portfolio
    return port_return


# ### Portfolio Volatility Function

# In[18]:


# Function takes a dataframe and a weight array
def volatility_cal(df, weight):
    # Generating the covariance matrix
    stock_cov = df.cov()
    # Generating the weight matrix
    weight_matrix = np.outer(weight, weight)
    # Scalar multiplication of both matrices, and then summing all elements to get the portfolio variance
    port_var = (stock_cov * weight_matrix).sum().sum()
    # Taking the standard deviation of the variance
    port_vol = np.sqrt(port_var)
    return port_vol


# ### Sharpe Ratio Function

# In[19]:


# Risk-free rate used is adjusted for by the input parameters
effective_rf = rf_adjuster(risk_free_rate, return_type)
# Function takes a dataframe and a weight array
def sharpe_ratio_cal(df, weight):
    # Sharpe ratio is the expected return minu the risk-free rate, divided by the standard deviation
    sharpe = (expected_return_cal(df, weight) - effective_rf)/volatility_cal(df, weight)
    return sharpe


# ### Optimisation using scipy.opt

# Before using the `opt.minimize` function, there are some things that need to be defined:

# In[20]:


# Defining the constarint, which is that all weights need to sum to 1
def weight_constraint(weight):
    return weight.sum() - 1

# Defining the starting state of the list in which to store the bounds
bnds = []

# The list of bounds should be the size of the number of stocks. Each iteration of the loop will store the extracted maximum portfolio weight as the upper limit and 0 as the lower limit
for i, row in stock_details_excel.iterrows():
    bnds.append((0, row["Max Portfolio Weight"]))

# Defining the constraint for the the minimise function
con = {"type":"eq", "fun":weight_constraint}


# Now the optimisation can be done

# In[21]:


# Initial guess value from inputted weights
x_0 = stock_weights_array

# Constrained maximisation of the Sharpe Ratio
result = opt.minimize(lambda x: -sharpe_ratio_cal(stock_returns, x),
                     constraints=con,
                     bounds=bnds,
                     x0=x_0)
print(result)


# Compiling the information into a pandas datadrame:

# In[22]:


# Adding the current weight of the stocks in the portfolio
stock_details_excel.insert(1, "Current Weight", x_0)
# Adding the optimal portfolio weight
stock_details_excel["Optimal Weight"] = result.x
# Getting the sum of the total value invested
portfolio_sum = stock_details_excel["Amount Invested"].sum()
# Getting the optimal value invested in each stock
stock_details_excel["Optimal Investment"] = round(stock_details_excel["Optimal Weight"] * portfolio_sum, 2)
# Rounding the optimal weights to 10 decimal places
stock_details_excel["Optimal Weight"] = round(stock_details_excel["Optimal Weight"], 10)
# Arranging the dataframe according to the original index for better tacking for the user
stock_details_excel = stock_details_excel.sort_index()
stock_details_excel


# One can also use the previously defined functions to get the optimal portfolio's expected return, volatility and sharpe ratio

# In[23]:


opt_return = expected_return_cal(stock_returns,result.x)
opt_volatility = volatility_cal(stock_returns, result.x)
opt_sharpe = -result.fun


# ## Writing the metrics into an excel output file

# These metrics can now be written into output excel file for easier viewing, and be compared to the original portfolio's metrics This can be done with pandas, but that means that the data first needs to be wrapped up in a dataframe

# In[24]:


# Storing the data to show in lists
metric_name = ['Sharpe Ratio', f'Portfolio Expected Return ({return_type})', 'Portfolio Volatility']
current_metric_value = [round(sharpe_ratio, 4), round(portfolio_expected_return, 4), round(portfolio_sd, 4)]
optimal_metric_value = [round(opt_sharpe, 4), round(opt_return, 4), round(opt_volatility, 4)]

# Creating the dataframe
metric_table = pd.DataFrame({'Metric Name': metric_name,
                             'Current Portfolio': current_metric_value,
                            'Optimal Portfolio': optimal_metric_value})
metric_table


# It would also be useful to write in a separate sheet the input details of what the user gave to the programme, so that it would be easier for the user to keep track of their excel outputs. For this the pandas dataframes already exist, which are the original input tables `stock_details_excel` and `other_paramters`. However, `other_parameters`'s datetimes need to be first turned into dates only. The writing of the metrics data to an excel file should only be done if the optimisation was successul

# In[25]:


# Only write the data to the excel file if the optimisation was successful
if result.success == True:
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
                             sheet_name='Stocks Details',
                             index=False)
        # Adding a sheet of the other parameters given by the useer
        other_parameters.to_excel(writer,
                             sheet_name='Inputted Misc.',
                             index=False)
else:
    error_df = pd.DataFrame({"ERROR": "OPTIMISATION WAS NOT SUCCESSFUL. CHECK INPUTS"},index=[1])
    error_df.to_excel("Excel_Outputs/ERROR.xlsx",
                     index=False)


# In[ ]:




