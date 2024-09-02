import numpy as np
import pytest

from traderlib import (
    computeBollingerBands,
    computeEMA,
    computeMACD,
    computeMovingAverageCrossover,
    computeResistance,
    computeRSI,
    computeSMA,
    computeStochasticOscillator,
    computeSupport,
)


# Helper function to generate test data
def generate_data(length, seed=42):
    np.random.seed(seed)
    return np.random.rand(length).astype(np.float32) * 100


@pytest.fixture
def price_data():
    return generate_data(1000)


@pytest.fixture
def short_price_data():
    return generate_data(10)


def test_computeEMA(price_data):
    lookback = 14
    result = computeEMA(price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(np.isfinite(result))


def test_computeStochasticOscillator(price_data):
    lookback = 14
    result = computeStochasticOscillator(
        price_data, price_data * 0.95, price_data * 1.05, lookback
    )
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(result >= 0) and np.all(result <= 100)


def test_computeResistance(price_data):
    lookback = 14
    result = computeResistance(price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(np.isfinite(result))


def test_computeSupport(price_data):
    lookback = 14
    result = computeSupport(price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(np.isfinite(result))


def test_computeSMA(price_data):
    lookback = 14
    result = computeSMA(price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(np.isfinite(result))


def test_computeBollingerBands(price_data):
    lookback = 14
    num_stddev = 2.0
    sma, upper_band, lower_band = computeBollingerBands(
        price_data, lookback, num_stddev
    )
    assert isinstance(sma, np.ndarray)
    assert isinstance(upper_band, np.ndarray)
    assert isinstance(lower_band, np.ndarray)
    assert sma.dtype == np.float32
    assert upper_band.dtype == np.float32
    assert lower_band.dtype == np.float32
    assert sma.shape == price_data.shape
    assert upper_band.shape == price_data.shape
    assert lower_band.shape == price_data.shape
    assert np.all(lower_band <= sma) and np.all(sma <= upper_band)


def test_computeRSI(price_data):
    lookback = 14
    result = computeRSI(price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(result >= 0) and np.all(result <= 100)


def test_computeMACD(price_data):
    fast_period = 12
    slow_period = 26
    signal_period = 9
    result = computeMACD(price_data, fast_period, slow_period, signal_period)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(np.isfinite(result))


def test_computeMovingAverageCrossover(price_data):
    short_lookback = 50
    long_lookback = 200
    result = computeMovingAverageCrossover(price_data, short_lookback, long_lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32
    assert result.shape == price_data.shape
    assert np.all(np.isin(result, [-1, 0, 1]))


# Edge cases and boundary condition tests
def test_short_lookback_computeEMA(short_price_data):
    lookback = 5
    result = computeEMA(short_price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == short_price_data.shape
    assert np.all(np.isfinite(result))


def test_zero_lookback_computeEMA(price_data):
    with pytest.raises(ValueError):
        computeEMA(price_data, 0)


def test_negative_lookback_computeEMA(price_data):
    with pytest.raises(ValueError):
        computeEMA(price_data, -1)


def test_nonfinite_data_computeEMA():
    data_array = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    with pytest.raises(ValueError):
        computeEMA(data_array, 14)


# Additional tests for boundary conditions and invalid inputs can be added similarly

if __name__ == "__main__":
    pytest.main()
