import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns



##########################################
# Heavy-tailed (fat-tailed) distribution #
##########################################

def plot_heavy_tail(X,
                    ax1, ax2,
                    real=False):
  """ Plot the heavy-tailed distribution of financial time series and compare with Gaussian """
  # Plot the distribution of financial time series
  if real:
    sns.kdeplot(X, ax=ax1, color='red', label='Real')
  else:
    sns.kdeplot(X, ax=ax1, label='Generated')
  # Plot the Gaussian distribution for comparison
  mu, sigma = np.mean(X), np.std(X)
  x = np.linspace(min(X), max(X), len(X))
  gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
  ax1.plot(x, gaussian, color='gray', linewidth=1, linestyle='--', label='Gaussian')
  ax1.set_xlabel('Returns')
  ax1.set_ylabel('Density')
  ax1.legend(loc='upper right')

  # Plot the Q-Q plot
  if real:
    sm.qqplot(X, line='s', marker='.', markeredgecolor='red', markerfacecolor='w', ax=ax2)
  else:
    sm.qqplot(X, line='s', marker='.', markerfacecolor='w', ax=ax2)
  ax2.yaxis.major.formatter.set_powerlimits((0,0))
  ax2.get_lines()[1].set_color("gray")
  ax2.get_lines()[1].set_linewidth("2")
  ax2.set_xlabel("Theoretical")
  ax2.set_ylabel("Sample")



################################################
# Decaying autocorrelation in absolute returns #
################################################

def plot_returns_autocorr(X, lags, ax, y_lim=-0.25, real=False):
  """ Plot the decaying autocorrelation in absolute returns """
  returns = pd.Series(X).dropna()
  if real:
    plot_acf(returns, lags=lags, ax=ax, color='red')
  else:
    plot_acf(returns, lags=lags, ax=ax)
  ax.set_ylim(y_lim, 1.0)
  ax.set_title("")
  ax.set_xlabel('Days')
  ax.set_ylabel('Autocorr.')
  if real:
    for line in ax.get_lines():               # Change marker and stemline color
      line.set_color('red')
    # Change the color of the confidence interval (the shaded area)
    for collection in ax.collections:
      collection.set_edgecolor('red')
      collection.set_facecolor('red')         # Change fill color
    ax.axhline(color='red', linestyle='--')   # Change the color of the horizontal line at y=0









