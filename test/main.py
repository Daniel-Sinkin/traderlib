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

# Ensure the results are numpy arrays and behave as such
assert isinstance(ema_result, np.ndarray), "EMA result is not a numpy array"
assert isinstance(
    stochastic_result, np.ndarray
), "Stochastic Oscillator result is not a numpy array"
assert isinstance(
    resistance_result, np.ndarray
), "Resistance result is not a numpy array"
assert isinstance(support_result, np.ndarray), "Support result is not a numpy array"

print("All tests passed!")
