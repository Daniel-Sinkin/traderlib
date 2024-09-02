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

def computeSMA(data_array: np.ndarray, lookback: int) -> np.ndarray: ...

"""
Compute the Simple Moving Average (SMA).

Parameters:
    data_array (numpy.ndarray): 1D array of float32 values representing the data.
    lookback (int): The number of periods to use in the calculation.

Returns:
    numpy.ndarray: A 1D array of float32 values representing the SMA.
"""

def computeBollingerBands(
    data_array: np.ndarray, lookback: int, num_stddev: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

"""
Compute the Bollinger Bands.

Parameters:
    data_array (numpy.ndarray): 1D array of float32 values representing the data.
    lookback (int): The number of periods to use in the calculation.
    num_stddev (float): The number of standard deviations to use for the bands.

Returns:
    tuple: A tuple of three numpy.ndarrays representing the SMA, upper band, and lower band.
"""

def computeRSI(data_array: np.ndarray, lookback: int) -> np.ndarray: ...

"""
Compute the Relative Strength Index (RSI).

Parameters:
    data_array (numpy.ndarray): 1D array of float32 values representing the data.
    lookback (int): The number of periods to use in the calculation.

Returns:
    numpy.ndarray: A 1D array of float32 values representing the RSI.
"""

def computeMACD(
    data_array: np.ndarray, fast_period: int, slow_period: int, signal_period: int
) -> np.ndarray: ...

"""
Compute the Moving Average Convergence Divergence (MACD).

Parameters:
    data_array (numpy.ndarray): 1D array of float32 values representing the data.
    fast_period (int): The period for the fast EMA.
    slow_period (int): The period for the slow EMA.
    signal_period (int): The period for the signal line.

Returns:
    numpy.ndarray: A 1D array of float32 values representing the MACD signal line.
"""

def computeMovingAverageCrossover(
    data_array: np.ndarray, short_lookback: int, long_lookback: int
) -> np.ndarray: ...

"""
Compute the Moving Average Crossover signals.

Parameters:
    data_array (numpy.ndarray): 1D array of float32 values representing the data.
    short_lookback (int): The lookback period for the short-term SMA.
    long_lookback (int): The lookback period for the long-term SMA.

Returns:
    numpy.ndarray: A 1D array of int32 values representing the signals (-1 for sell, 0 for neutral, 1 for buy).
"""
