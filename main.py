import numpy as np
import traderlib

# Create sample numpy arrays
data_array = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float32)
low_prices = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.float32)
high_prices = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.float32)
close_prices = np.array(
    [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5], dtype=np.float32
)

lookback = 3

# Compute Exponential Moving Average (EMA)
ema_result = traderlib.computeEMA(data_array, lookback)
print("Exponential Moving Average (EMA):")
print(ema_result)

# Compute Stochastic Oscillator
stochastic_result = traderlib.computeStochasticOscillator(
    close_prices, low_prices, high_prices, lookback
)
print("Stochastic Oscillator:")
print(stochastic_result)

# Compute Resistance Level
resistance_result = traderlib.computeResistance(high_prices, lookback)
print("Resistance Level:")
print(resistance_result)

# Compute Support Level
support_result = traderlib.computeSupport(low_prices, lookback)
print("Support Level:")
print(support_result)

# Compute Simple Moving Average (SMA)
sma_result = traderlib.computeSMA(data_array, lookback)
print("Simple Moving Average (SMA):")
print(sma_result)

# Compute Bollinger Bands
bollinger_sma, upper_band, lower_band = traderlib.computeBollingerBands(
    data_array, lookback, 2.0
)
print("Bollinger Bands (SMA, Upper Band, Lower Band):")
print(bollinger_sma)
print(upper_band)
print(lower_band)

# Compute Relative Strength Index (RSI)
rsi_result = traderlib.computeRSI(data_array, lookback)
print("Relative Strength Index (RSI):")
print(rsi_result)

# Compute Moving Average Convergence Divergence (MACD)
macd_result = traderlib.computeMACD(
    data_array, fast_period=2, slow_period=5, signal_period=3
)
print("Moving Average Convergence Divergence (MACD):")
print(macd_result)

# Compute Moving Average Crossover signals
crossover_result = traderlib.computeMovingAverageCrossover(
    data_array, short_lookback=2, long_lookback=5
)
print("Moving Average Crossover signals:")
print(crossover_result)

# Ensure the results are numpy arrays and behave as such
assert isinstance(ema_result, np.ndarray), "EMA result is not a numpy array"
assert isinstance(
    stochastic_result, np.ndarray
), "Stochastic Oscillator result is not a numpy array"
assert isinstance(
    resistance_result, np.ndarray
), "Resistance result is not a numpy array"
assert isinstance(support_result, np.ndarray), "Support result is not a numpy array"
assert isinstance(sma_result, np.ndarray), "SMA result is not a numpy array"
assert isinstance(bollinger_sma, np.ndarray), "Bollinger SMA is not a numpy array"
assert isinstance(upper_band, np.ndarray), "Bollinger upper band is not a numpy array"
assert isinstance(lower_band, np.ndarray), "Bollinger lower band is not a numpy array"
assert isinstance(rsi_result, np.ndarray), "RSI result is not a numpy array"
assert isinstance(macd_result, np.ndarray), "MACD result is not a numpy array"
assert isinstance(
    crossover_result, np.ndarray
), "Crossover signals result is not a numpy array"

print("All tests passed!")
