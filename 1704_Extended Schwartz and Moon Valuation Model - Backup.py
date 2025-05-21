# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.integrate import simps
import os
plt.style.use('ggplot')

np.random.seed(10)

data_folder = r'C:\Users\phili\OneDrive\Dokumente\0425_Thesis\425_JupyterCode\425_v3_JupyterCode'

df = pd.read_csv(os.path.join(data_folder, "v3_input_params_data.csv"))

df = df.set_index('Unnamed: 0')

df.index.name = ''

df.columns


# %% [markdown]
# # Monte Carlo - Extended Schwartz/Moon model

# %%
def schwarz_moon_firm_values_simulation(p, n_paths, T, dt):
    values = np.zeros(n_paths)

    for i in range(n_paths):
        # Initial values
        R = p.R0
        mu = p.mu0
        sigma = p.sigma0
        eta = p.eta0
        g = p.g0
        omega = p.omega0
        CR = p.CR0
        PPE = p.PPE0
        L = p.L0
        X = p.X0
        r = p.r0
        D = 0  # Total discount factor

        for t in range(T):
            # Generate standard normals
            z_R = np.random.normal()
            z_mu = np.random.normal()
            z_g = np.random.normal()
            z_r = np.random.normal()

            # Risk-neutral drifts
            mu_adj = mu - p.lambda_R * sigma
            mu_mu_adj = p.k_mu * (p.mu_bar - mu) - p.lambda_mu * eta
            mu_g_adj = p.k_g * (p.g_bar - g) - p.lambda_g * omega

            # Update variables
            R *= np.exp((mu_adj - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z_R)
            mu += mu_mu_adj * dt + eta * np.sqrt(dt) * z_mu
            g += mu_g_adj * dt + omega * np.sqrt(dt) * z_g
            sigma += p.k_sigma * (p.sigma_bar - sigma) * dt
            eta *= np.exp(-p.k_eta * dt)
            omega += p.k_omega * (p.omega_bar - omega) * dt
            r += p.k_r * (p.r_bar - r) * dt + p.v * np.sqrt(dt) * z_r
            r = max(r, 0.00001)
            CR += p.k_CR * (p.CR_bar - CR) * dt

            # Costs, EBITDA, CapEx
            cost = g * R + p.F
            EBITDA = max(R - cost, 0)
            Dep = p.DR * PPE
            CapEx = CR * R
            PPE += (CapEx - Dep) * dt
            taxable_income = R - cost - Dep
            tax = p.tau * max(taxable_income - L, 0)
            L = max(L - (taxable_income - tax), 0)
            Y = R - cost - Dep - tax
            FCF = Y + Dep - CapEx

            # Cash flow accumulation
            X = X * (1 + r * dt) + FCF * dt
            D += r * dt

        # Terminal firm value
        CV = p.M * max(R - (g * R + p.F), 0)
        V = (X + CV) * np.exp(-D)
        values[i] = V
    return values


def plot_firm_value_distribution(values):
    # Compute basic statistics
    median_val = np.median(values)
    mean_val = np.mean(values)
    
    # Use the 1st and 99th percentiles to define the x-axis range (in million USD)
    lower_bound = np.percentile(values, 0)
    upper_bound = np.percentile(values, 95)
    
    # Create 50 bins in this range
    bins = np.linspace(lower_bound, upper_bound, num=50)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 4.5))
    n, bins, patches = ax.hist(values, bins=bins, edgecolor='black', color='#5DADE2')
    
    # Draw vertical lines for the median and mean values
    ax.axvline(median_val, color='black', linestyle='-', linewidth=1.5)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5)
    
    # Use the top of the y-axis for positioning text annotations
    y_max = ax.get_ylim()[1]
    ax.text(median_val, y_max * 0.85, f"Median = {median_val/1e3:.2f}B", 
            rotation=90, va='center', ha='right', fontsize=9, fontweight='bold')
    ax.text(mean_val, y_max * 0.85, f"Mean = {mean_val/1e3:.2f}B", 
            rotation=90, va='center', ha='left', fontsize=9, fontweight='bold')
    
    # Set title and labels
    ax.set_title("Firm Value Distribution", fontsize=12, fontweight='bold')
    ax.set_xlabel("Firm Value (in million USD)")
    ax.set_ylabel("Frequency")
    
    # Format the x-axis ticks with commas
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    
    # Add y-axis gridlines for clarity
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    ax.xaxis.grid(False)
    
    plt.tight_layout()
    plt.show()


def compute_simulation_stats(
    firm_values,
    debt,
    shares,
    dt=1,
    horizon=100,
    n_sims=None):
    
    if n_sims is None:
        n_sims = len(firm_values)
    
    min_val = np.min(firm_values)
    q25 = np.percentile(firm_values, 25)
    median_val = np.median(firm_values)
    mean_val = np.mean(firm_values)
    q75 = np.percentile(firm_values, 75)
    max_val = np.max(firm_values)
    std_val = np.std(firm_values, ddof=1)
    
    p_insolvency = len(firm_values[firm_values <= 0])/len(firm_values)

    equity_values = firm_values - debt

    stock_prices = (equity_values * 1e6) / shares
    
    mean_stock_price = np.mean(stock_prices)
    std_stock_price = np.std(stock_prices, ddof=1)
    
    results = {
        "Delta t": dt,
        "Zeitraum": horizon,
        "Simulationseinheiten": n_sims,
        "Minimum": min_val,
        "25%": q25,
        "Median": median_val,
        "Mittelwert": mean_val,
        "75%": q75,
        "Maximum": max_val,
        "Std. V0": std_val,
        "Aktienpreis": mean_stock_price,
        "Std. Aktienpreis": std_stock_price,
        "Wahrscheinlichkeit Insolvenz": p_insolvency
    }
    
    return pd.DataFrame([results])


def simulate_firm_value(p, n_paths, T, dt):
    """
    Run Monte Carlo simulation and return average firm value.
    p: parameter Series (original input parameters)
    n_paths: number of simulation paths
    T: time horizon (quarters)
    dt: time increment (quarter)
    """
    values = np.zeros(n_paths)
    for i in range(n_paths):
        R = p.R0
        mu = p.mu0
        sigma = p.sigma0
        eta = p.eta0
        g = p.g0
        omega = p.omega0
        CR = p.CR0
        PPE = p.PPE0
        L = p.L0
        X = p.X0
        r = p.r0
        D = 0  # Cumulative discount factor
        
        for t in range(T):
            z_R = np.random.normal()
            z_mu = np.random.normal()
            z_g = np.random.normal()
            z_r = np.random.normal()
            
            mu_adj = mu - p.lambda_R * sigma
            mu_mu_adj = p.k_mu * (p.mu_bar - mu) - p.lambda_mu * eta
            mu_g_adj = p.k_g * (p.g_bar - g) - p.lambda_g * omega
            
            R *= np.exp((mu_adj - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_R)
            mu += mu_mu_adj * dt + eta * np.sqrt(dt) * z_mu
            g += mu_g_adj * dt + omega * np.sqrt(dt) * z_g
            sigma += p.k_sigma * (p.sigma_bar - sigma) * dt
            eta *= np.exp(-p.k_eta * dt)
            omega += p.get("k_w", 0.09) * (p.omega_bar - omega) * dt
            r += p.k_r * (p.r_bar - r) * dt + p.v * np.sqrt(dt) * z_r
            r = max(r, 0.00001)
            CR += p.get("k_CR", 0.09) * (p.CR_bar - CR) * dt
            
            cost = g * R + p.F
            EBITDA = max(R - cost, 0)
            Dep = p.DR * PPE
            CapEx = CR * R
            PPE += (CapEx - Dep) * dt
            taxable_income = R - cost - Dep
            tax = p.tau * max(taxable_income - L, 0)
            L = max(L - (taxable_income - tax), 0)
            Y = R - cost - Dep - tax
            FCF = Y + Dep - CapEx
            
            X = X * (1 + r * dt) + FCF * dt
            D += r * dt
            
        CV = p.M * max(R - (g * R + p.F), 0)
        values[i] = (X + CV) * np.exp(-D)
    return np.mean(values)


# %% [markdown]
# # Explore df

# %%
df

# %%
p = df.loc['fb_2012']

n_paths = 100000

T = int(p['T'])

dt = p['dt']

values = schwarz_moon_firm_values_simulation(p=p, 
                                             n_paths=n_paths,
                                             T=T,
                                             dt=dt)

# %% [markdown]
# # Figure 3. 

# %%
plot_firm_value_distribution(values)

# %%
reasonable_values = values[values < np.percentile(values, 90)]

# %%
equity_value = np.mean(values) - p.debt  # in million USD
total_shares = p.stocknumout + p.stocknumcb
price_per_share = equity_value * 1e6 / total_shares  # convert million USD to USD per share

result_df = pd.DataFrame({
    "Equity Value (M USD)": [equity_value],
    "Total Shares": [total_shares],
    "Price per Share (USD)": [price_per_share]
})

print(f'Share Price: ${result_df.round(2).iloc[0,2]}')

result_df

# %%
result_df.to_csv(os.path.join(data_folder, 'share_price.csv'))

# %%
df_stats = compute_simulation_stats(
        firm_values=values,
        debt=p.debt,
        shares=p.stocknumout,
        dt=1,
        horizon=100,
        n_sims=n_paths)

df_stats

# %%
df_stats.to_csv(os.path.join(data_folder, 'descriptive_stats_of_values.csv'))

# %% [markdown]
# # Sensitivity Analysis

# %%
n_paths_sens = 1000  # moderate number of paths for sensitivity analysis
T_base = int(p['T'])
dt = p['dt']

# Calculate the base case firm value (in million USD)
base_value = simulate_firm_value(p, n_paths_sens, T_base, dt)

# List of parameter keys to test (one-at-a-time)
sensitivity_keys = ["mu0", "sigma0", "eta0", "g0", "omega0", "CR0", "M", "v", "k_mu", "T"]

sensitivity_results = []

for key in sensitivity_keys:
    p_new = p.copy()
    if key == "T":
        new_T = int(np.round(T_base * 1.1))
        p_new["T"] = new_T
        new_value = simulate_firm_value(p_new, n_paths_sens, new_T, dt)
        original_val = T_base
        new_val = new_T
    else:
        p_new[key] = p_new[key] * 1.1
        new_value = simulate_firm_value(p_new, n_paths_sens, T_base, dt)
        original_val = p[key]
        new_val = p_new[key]
        
    rel_diff = (new_value - base_value) / base_value * 100
    sensitivity_results.append({
        "Parameter": key,
        "Original Value": original_val,
        "Sensitivity Test Value": new_val,
        "Resulting Firm Value (M USD)": new_value,
        "Relative Difference (%)": rel_diff
    })

sensitivity_df = pd.DataFrame(sensitivity_results)

sensitivity_df

# %%
sensitivity_df.to_csv(os.path.join(data_folder, 'sensitivity_analysis.csv'))

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
#

# %% [markdown]
# # API Parameters Estimation

# %%
import functools as ft
import datetime as dt
import requests
import json

FMP_API_KEY = '8b2ebb502832761e2fb2642e59509f22'


def get_treasury_yields(start_date, end_date):
    session = requests.Session()
    request = f"https://financialmodelingprep.com/api/v4/treasury?\
                from={start_date}&to={end_date}&apikey={FMP_API_KEY}".replace(" ", "")
    r = session.get(request)
    if r.status_code == requests.codes.ok:
        try:
            df = pd.DataFrame(json.loads(r.text))
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            return df
        except: print(f'No Data')
    else: return
    
    
def get_symbol_daily_prices(symbol, start_date, end_date):
    session = requests.Session()
    request = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}\
                # ?from={start_date}&to={end_date}&apikey={FMP_API_KEY}".replace(" ", "")
    r = session.get(request)
    if r.status_code == requests.codes.ok:
        try:
            df = pd.DataFrame(json.loads(r.text)['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df['symbol'] = symbol
            return df[['symbol', 'open', 'high', 'low', 'close', 'volume', 'adjClose']]
        except: print(f'{symbol} No Data')
    else: return
    
    
def get_quarterly_income_statement(symbol):
    session = requests.Session()
    url = f'https://financialmodelingprep.com//api/v3/income-statement/\
            {symbol}?period=quarter&limit=80&apikey={FMP_API_KEY}'.replace(' ', '')
    r = session.get(url)
    if r.status_code == requests.codes.ok:
        try:
            df = pd.DataFrame(json.loads(r.text))
            df['fillingDate'] = pd.to_datetime(df['fillingDate'])
            df = df.set_index('fillingDate').sort_index().drop_duplicates()
            return df
        except:
            print('Error')
            
            
def get_quarterly_balance_sheet_statement(symbol):
    session = requests.Session()
    url = f'https://financialmodelingprep.com//api/v3/balance-sheet-statement/\
            {symbol}?period=quarter&limit=80&apikey={FMP_API_KEY}'.replace(' ', '')
    r = session.get(url)
    if r.status_code == requests.codes.ok:
        try:
            df = pd.DataFrame(json.loads(r.text))
            df['fillingDate'] = pd.to_datetime(df['fillingDate'])
            df = df.set_index('fillingDate').sort_index().drop_duplicates()
            return df
        except:
            print('Error')
    

def get_quarterly_cash_flow_statement(symbol):
    session = requests.Session()
    url = f'https://financialmodelingprep.com//api/v3/cash-flow-statement/\
            {symbol}?period=quarter&limit=80&apikey={FMP_API_KEY}'.replace(' ', '')
    r = session.get(url)
    if r.status_code == requests.codes.ok:
        try:
            df = pd.DataFrame(json.loads(r.text))
            df['fillingDate'] = pd.to_datetime(df['fillingDate'])
            df = df.set_index('fillingDate').sort_index().drop_duplicates()
            return df
        except:
            print('Error')
            
def get_all_yields():
    yields_df = pd.DataFrame()
    end_date = dt.date.today()
    for i in range(3*25):
        start_date = end_date-pd.DateOffset(90)
        temp_yields = get_treasury_yields(start_date=start_date.strftime('%Y-%m-%d'),
                                          end_date=end_date.strftime('%Y-%m-%d'))
        yields_df = pd.concat([yields_df, temp_yields], axis=0)
        end_date = start_date
        yields_df = yields_df.drop_duplicates().sort_index()
    return yields_df


# %%
yields_df = get_all_yields()

yields_df = yields_df.groupby(pd.Grouper(freq='QE')).last()


# %%
SYMBOL = 'META'

income_statement = get_quarterly_income_statement(symbol=SYMBOL)

balance_sheet_statement = get_quarterly_balance_sheet_statement(symbol=SYMBOL)

cash_flow_statement = get_quarterly_cash_flow_statement(symbol=SYMBOL)

dataframes = [income_statement, balance_sheet_statement, cash_flow_statement]

fundamentals_data_df = ft.reduce(lambda left,right: pd.merge(left,
                                                             right,
                                                             left_index=True,
                                                             right_index=True,
                                                             suffixes=('', '_right')), dataframes)

fundamentals_data_df = fundamentals_data_df.iloc[-15*4:, :].drop(['date', 'reportedCurrency',
                                                                  'cik', 'acceptedDate',
                                                                  'calendarYear', 'period'], axis=1)

fundamentals_data_df = fundamentals_data_df.groupby(pd.Grouper(freq='QE')).last()

yields_df = yields_df.groupby(pd.Grouper(freq='QE')).last()

returns_df = pd.concat([get_symbol_daily_prices(symbol='META',
                            start_date=fundamentals_data_df.index.min().strftime('%Y-%m-%d'),
                            end_date=fundamentals_data_df.index.max().strftime('%Y-%m-%d'))\
                .groupby(pd.Grouper(freq='QE')).last()['adjClose'].pct_change().to_frame('meta_return'),

                        get_symbol_daily_prices(symbol='SPY',
                            start_date=fundamentals_data_df.index.min().strftime('%Y-%m-%d'),
                            end_date=fundamentals_data_df.index.max().strftime('%Y-%m-%d'))\
                .groupby(pd.Grouper(freq='QE')).last()['adjClose'].pct_change().to_frame('spy_return')],
                       axis=1).dropna()

# %%
final_df = ft.reduce(lambda left,right: pd.merge(left,
                                                 right,
                                                 left_index=True,
                                                 right_index=True,
                                                 suffixes=('', '_right')), [fundamentals_data_df,
                                                                            yields_df[['month3']],
                                                                            returns_df])

final_df.columns.tolist()

# %%
gg = final_df[['revenue', 'costOfRevenue', 'generalAndAdministrativeExpenses',
          'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses',
          'researchAndDevelopmentExpenses', 'otherExpenses', 'spy_return', 'month3',
          'cashAndCashEquivalents', 'propertyPlantEquipmentNet', 'totalDebt',
          'weightedAverageShsOutDil']]

gg['fixed_cost'] = gg['generalAndAdministrativeExpenses']+gg['sellingAndMarketingExpenses']+\
                   gg['sellingGeneralAndAdministrativeExpenses']

gg['variable_cost'] = gg['researchAndDevelopmentExpenses']+gg['otherExpenses']

gg['risk_free_rate'] = gg['month3']

gg['stocknumout'] = gg['weightedAverageShsOutDil']

gg['revenue_growth'] = gg['revenue'].pct_change()

raw_df = gg[['revenue', 'revenue_growth', 'costOfRevenue', 'fixed_cost', 'variable_cost', 'risk_free_rate', 
            'spy_return', 'cashAndCashEquivalents', 'propertyPlantEquipmentNet', 'totalDebt',
            'weightedAverageShsOutDil']]

raw_df

# %%
import pandas as pd
import numpy as np

def estimate_simulation_params(raw_df,
                               target_quarter, 
                               window=6):
    if not np.issubdtype(raw_df.index.dtype, np.datetime64):
        try:
            raw_df.index = pd.to_datetime(raw_df.index)
        except Exception as e:
            raise ValueError("Could not convert DataFrame index to datetime. Ensure your index or quarter column is parseable.") 
    
    target_dt = pd.to_datetime(target_quarter)
    
    # Select all rows up to and including the target quarter
    df_hist = raw_df[raw_df.index <= target_dt].sort_index()
    if df_hist.empty:
        raise ValueError("No data available for quarters <= target_quarter.")
    
    # Use the last 'window' quarters for estimation
    df_window = df_hist.tail(window)
    
    # R0: Revenue in target quarter (convert from dollars to millions)
    R0 = df_hist.loc[target_dt, "revenue"] / 1e6
    
    # mu0: Average revenue growth (as decimal) over the window
    mu0 = df_window["revenue_growth"].dropna().mean()
    
    # sigma0: Std dev of revenue growth over the window
    sigma0 = df_window["revenue_growth"].dropna().std(ddof=0)
    
    # g0: Average variable cost fraction over the window = (variable_cost / revenue)
    g0 = (df_window["variable_cost"] / df_window["revenue"]).dropna().mean()
    # g_bar: Assume long-term variable cost fraction is 10% higher than g0
    g_bar = g0 * 1.1
    
    # r0: Risk-free rate for the target quarter
    r0 = df_hist.loc[target_dt, "risk_free_rate"]/100
    
    # r_bar: Average risk-free rate over the window
    r_bar = df_window["risk_free_rate"].dropna().mean()/100
    
    sigma_M = df_window["spy_return"].std(ddof=0)
    
    # Use "revenue_growth" as the revenue shock proxy.
    corr_R = df_window["revenue_growth"].corr(df_window["spy_return"])

    # Approximate the shock in expected growth by taking the first difference of revenue_growth.
    rev_growth_diff = df_window["revenue_growth"].diff().dropna()
    # Align spy_return by dropping the first row (so they have the same index)
    spy_return_aligned = df_window["spy_return"].iloc[1:]
    corr_mu = rev_growth_diff.corr(spy_return_aligned)
    
    # Compute variable cost fraction: g = variable_cost / revenue.
    df_for_g = df_window.copy()
    df_for_g["g"] = df_for_g["variable_cost"] / df_for_g["revenue"]
    g_diff = df_for_g["g"].diff().dropna()
    spy_return_aligned2 = df_for_g["spy_return"].iloc[1:]
    corr_g = g_diff.corr(spy_return_aligned2)
    
    # F: Fixed cost for target quarter (convert to millions)
    F = df_hist.loc[target_dt, "fixed_cost"] / 1e6
    
    # X0: Cash and cash equivalents for target quarter (convert to millions)
    X0 = df_hist.loc[target_dt, "cashAndCashEquivalents"] / 1e6
    
    # PPE0: Property, plant, and equipment (convert to millions)
    PPE0 = df_hist.loc[target_dt, "propertyPlantEquipmentNet"] / 1e6
    
    # debt: Total debt for target quarter (convert to millions)
    debt = df_hist.loc[target_dt, "totalDebt"] / 1e6
    
    # stocknumout: Weighted average shares outstanding (fully diluted)
    stocknumout = df_hist.loc[target_dt, "weightedAverageShsOutDil"]
    
    # stocknumcb: Assume treasury shares = 0 (if not provided)
    stocknumcb = 0
    
    # --- Fixed / Assumed (Literature) Values ---
    L0 = 1620            # Initial loss carryforward (million USD) [assumed]
    eta0 = 0.03          # Volatility of the growth rate
    mu_bar = raw_df["revenue_growth"].loc[:target_quarter].dropna().mean()      # Long-term average revenue growth rate
    sigma_bar =  raw_df["revenue_growth"].loc[:target_quarter].dropna().std(ddof=0)    # Long-term volatility of revenue growth
    omega0 = 0.0655      # Initial volatility of variable costs
    omega_bar = raw_df["variable_cost"].loc[:target_quarter].dropna().pct_change().std(ddof=0)    # Long-term volatility of variable costs
    DR = 0.1038          # Depreciation ratio
    CR0 = 0.3237         # Initial capital expenditure ratio
    CR_bar = 0.1038      # Long-term capital expenditure ratio
    tau = 0.35           # Corporate tax rate
    M = 10               # EBITDA multiple for terminal value
    lambda_R = corr_R * sigma_M    # Market price of risk for revenues
    lambda_mu = corr_mu * sigma_M   # Market price of risk for expected growth
    lambda_g = corr_g * sigma_M    # Market price of risk for variable costs
    v = raw_df["risk_free_rate"].loc[:target_quarter].dropna().div(100).std(ddof=0)    # Volatility of the risk-free rate
    k_mu = 0.09          # Mean reversion coefficient for mu
    k_sigma = 0.09       # Mean reversion coefficient for sigma
    k_eta = 0.09         # Mean reversion coefficient for eta
    k_g = 0.09           # Mean reversion coefficient for g
    k_omega = 0.09       # Mean reversion coefficient for k_omega
    k_w = 0.09           # Mean reversion coefficient for ω (variable cost volatility)
    k_r = 0.09           # Mean reversion coefficient for risk-free rate
    k_CR = 0.09          # Mean reversion coefficient for CR
    dt = 1               # Time increment (1 quarter)
    T = 100              # Time horizon (100 quarters, i.e., 25 years)
    
    params_est = {
        "X0": X0,
        "L0": L0,
        "PPE0": PPE0,
        "R0": R0,
        "mu0": mu0,
        "sigma0": sigma0,
        "eta0": eta0,
        "mu_bar": mu_bar,
        "sigma_bar": sigma_bar,
        "F": F,
        "g0": g0,
        "g_bar": g_bar,
        "omega0": omega0,
        "omega_bar": omega_bar,
        "DR": DR,
        "CR0": CR0,
        "CR_bar": CR_bar,
        "tau": tau,
        "M": M,
        "lambda_R": lambda_R,
        "lambda_mu": lambda_mu,
        "lambda_g": lambda_g,
        "r0": r0,
        "r_bar": r_bar,
        "v": v,
        "k_mu": k_mu,
        "k_sigma": k_sigma,
        "k_eta": k_eta,
        "k_omega": k_omega,
        "k_g": k_g,
        "k_w": k_w,
        "k_r": k_r,
        "k_CR": k_CR,
        "dt": dt,
        "T": T,
        "debt": debt,
        "stocknumcb": stocknumcb,
        "stocknumout": stocknumout
    }
    return pd.DataFrame([params_est])



# %%
start_date_quarter = "2024-12-31"

estimated_params = estimate_simulation_params(raw_df=raw_df,
                                              target_quarter=start_date_quarter,
                                              window=6)

estimated_params = estimated_params.T

estimated_params.columns = [f'fb_{start_date_quarter}']

estimated_params = estimated_params.T

estimated_params

# %%
df = pd.concat([df, estimated_params], axis=0)

# %%
df

# %%
df.to_csv(os.path.join(data_folder, 'input_params_data.csv'))

# %%
df.to_csv('D:/data_folder/maphil_/input_params_data.csv')

# %%

# %%

# %%

# %%

# %%

# %%
