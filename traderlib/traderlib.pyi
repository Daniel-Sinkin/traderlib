import numpy as np

def computeEMA(data_array: np.ndarray, lookback: int) -> np.ndarray: ...

"""
Compute the Exponential Moving Average (EMA).

Parameters:
    data_array (numpy.ndarray): 1D array of float32 values representing the data.
    lookback (int): The number of periods to use in the calculation.

Returns:
    numpy.ndarray: A 1D array of float32 values representing the EMA.
"""

def computeStochasticOscillator(
    close_prices: np.ndarray,
    low_prices: np.ndarray,
    high_prices: np.ndarray,
    lookback: int,
) -> np.ndarray: ...

"""
Compute the Stochastic Oscillator.

Parameters:
    close_prices (numpy.ndarray): 1D array of float32 values representing the closing prices.
    low_prices (numpy.ndarray): 1D array of float32 values representing the low prices.
    high_prices (numpy.ndarray): 1D array of float32 values representing the high prices.
    lookback (int): The number of periods to use in the calculation.

Returns:
    numpy.ndarray: A 1D array of float32 values representing the Stochastic Oscillator.
"""

def computeResistance(high_prices: np.ndarray, lookback: int) -> np.ndarray: ...

"""
Compute the Resistance Level.

Parameters:
    high_prices (numpy.ndarray): 1D array of float32 values representing the high prices.
    lookback (int): The number of periods to use in the calculation.

Returns:
    numpy.ndarray: A 1D array of float32 values representing the resistance levels.
"""

def computeSupport(low_prices: np.ndarray, lookback: int) -> np.ndarray: ...

"""
Compute the Support Level.

Parameters:
    low_prices (numpy.ndarray): 1D array of float32 values representing the low prices.
    lookback (int): The number of periods to use in the calculation.

Returns:
    numpy.ndarray: A 1D array of float32 values representing the support levels.
"""
