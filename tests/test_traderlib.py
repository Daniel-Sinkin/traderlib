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


def generate_data(length, seed=42):
    _rng = np.random.default_rng(seed)
    return _rng.uniform(0, 100.0, length).astype(np.float32)


@pytest.fixture
def price_data():
    return generate_data(1_000_000)


@pytest.fixture
def short_price_data():
    return generate_data(10)


@pytest.fixture
def single_element_data():
    return np.array([42.0], dtype=np.float32)


# Test for computeEMA
def test_computeEMA(price_data):
    lookback = 14
    result = computeEMA(price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(np.isfinite(result))


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


def test_empty_array_computeEMA():
    data_array = np.array([], dtype=np.float32)
    assert len(data_array) == 0


def test_single_element_computeEMA(single_element_data):
    result = computeEMA(single_element_data, 1)
    assert isinstance(result, np.ndarray)
    assert result.shape == single_element_data.shape
    assert result[0] == single_element_data[0]


def test_max_float_computeEMA(price_data):
    price_data[0] = np.finfo(np.float32).max
    result = computeEMA(price_data, 14)
    assert np.isfinite(result).all()


def test_min_float_computeEMA(price_data):
    price_data[0] = np.finfo(np.float32).min
    result = computeEMA(price_data, 14)
    assert np.isfinite(result).all()


def test_random_nan_middle_computeEMA(price_data):
    price_data[500] = np.nan
    with pytest.raises(ValueError):
        computeEMA(price_data, 14)


def test_large_lookback_computeEMA(short_price_data):
    lookback = 20
    result = computeEMA(short_price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.shape == short_price_data.shape
    assert np.all(np.isfinite(result))


def test_lookback_of_one_computeEMA(price_data):
    result = computeEMA(price_data, 1)
    assert isinstance(result, np.ndarray)
    assert result.shape == price_data.shape
    assert np.allclose(result, price_data, atol=5e-5)


# Test for computeStochasticOscillator
def test_computeStochasticOscillator(price_data):
    lookback = 14
    result = computeStochasticOscillator(
        price_data, price_data * 0.95, price_data * 1.05, lookback
    )
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(result >= 0) and np.all(result <= 100)


# Test for computeResistance
def test_computeResistance(price_data):
    lookback = 14
    result = computeResistance(price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(np.isfinite(result))


# Test for computeSupport
def test_computeSupport(price_data):
    lookback = 14
    result = computeSupport(price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(np.isfinite(result))


# Test for computeSMA
def test_computeSMA(price_data):
    lookback = 14
    result = computeSMA(price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(np.isfinite(result))


def test_empty_array_computeSMA():
    data_array = np.array([], dtype=np.float32)
    assert len(data_array) == 0


def test_single_element_computeSMA(single_element_data):
    result = computeSMA(single_element_data, 1)
    assert isinstance(result, np.ndarray)
    assert result.shape == single_element_data.shape
    assert result[0] == single_element_data[0]


# Test for computeBollingerBands
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


# Test for computeRSI
def test_computeRSI(price_data):
    lookback = 14
    result = computeRSI(price_data, lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(result >= 0) and np.all(result <= 100)


# Test for computeMACD
def test_computeMACD(price_data):
    fast_period = 12
    slow_period = 26
    signal_period = 9
    result = computeMACD(price_data, fast_period, slow_period, signal_period)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == price_data.shape
    assert np.all(np.isfinite(result))


# Test for computeMovingAverageCrossover
def test_computeMovingAverageCrossover(price_data):
    short_lookback = 50
    long_lookback = 200
    result = computeMovingAverageCrossover(price_data, short_lookback, long_lookback)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32
    assert result.shape == price_data.shape
    assert np.all(np.isin(result, [-1, 0, 1]))


if __name__ == "__main__":
    pytest.main()
