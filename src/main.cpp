#include <algorithm>
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

pybind11::array_t<float> computeEMA(const pybind11::array_t<float> &data_array, int lookback) {
    if (lookback <= 0) {
        throw std::invalid_argument("Lookback must be a positive integer.");
    }

    pybind11::buffer_info buf = data_array.request();

    if (buf.ndim != 1) {
        throw std::runtime_error("Input data should be a 1D array.");
    }

    const float *data_ptr = static_cast<const float *>(buf.ptr);
    size_t size = buf.size;

    // Special case for lookback of 1: return the input array unchanged
    if (lookback == 1) {
        // Return a copy of the input array
        pybind11::array_t<float> result(size);
        auto result_buf = result.request();
        float *result_ptr = static_cast<float *>(result_buf.ptr);
        std::copy(data_ptr, data_ptr + size, result_ptr);
        return result;
    }

    // Check for non-finite values (NaN, inf, -inf)
    for (size_t i = 0; i < size; ++i) {
        if (!std::isfinite(data_ptr[i])) {
            throw std::invalid_argument("Data array contains non-finite values (NaN, inf, -inf).");
        }
    }

    pybind11::array_t<float> ema(size);
    auto ema_buf = ema.request();
    float *ema_ptr = static_cast<float *>(ema_buf.ptr);

    float multiplier = 2.0f / (lookback + 1);
    ema_ptr[0] = data_ptr[0]; // Initialize the first value of EMA

    for (size_t i = 1; i < size; ++i) {
        ema_ptr[i] = ((data_ptr[i] - ema_ptr[i - 1]) * multiplier) + ema_ptr[i - 1];
    }

    return ema;
}

// Function to compute Stochastic Oscillator
pybind11::array_t<float> computeStochasticOscillator(const pybind11::array_t<float> &close_prices,
                                                     const pybind11::array_t<float> &low_prices,
                                                     const pybind11::array_t<float> &high_prices,
                                                     int lookback) {
    if (lookback <= 0) {
        throw std::invalid_argument("Lookback must be a positive integer.");
    }

    pybind11::buffer_info close_buf = close_prices.request();
    pybind11::buffer_info low_buf = low_prices.request();
    pybind11::buffer_info high_buf = high_prices.request();

    if (close_buf.ndim != 1 || low_buf.ndim != 1 || high_buf.ndim != 1) {
        throw std::runtime_error("Input data should be 1D arrays.");
    }

    if (close_buf.size != low_buf.size || close_buf.size != high_buf.size) {
        throw std::runtime_error("All input arrays must have the same size.");
    }

    const float *close_ptr = static_cast<const float *>(close_buf.ptr);
    const float *low_ptr = static_cast<const float *>(low_buf.ptr);
    const float *high_ptr = static_cast<const float *>(high_buf.ptr);
    size_t size = close_buf.size;

    pybind11::array_t<float> stochastic(size);
    auto stochastic_buf = stochastic.request();
    float *stochastic_ptr = static_cast<float *>(stochastic_buf.ptr);

    for (size_t i = 0; i < size; ++i) {
        if (i < lookback - 1) {
            stochastic_ptr[i] = 0.0f;
        } else {
            float highest_high = high_ptr[i];
            float lowest_low = low_ptr[i];
            for (int j = 0; j < lookback; ++j) {
                highest_high = std::max(highest_high, high_ptr[i - j]);
                lowest_low = std::min(lowest_low, low_ptr[i - j]);
            }
            stochastic_ptr[i] = 100.0f * (close_ptr[i] - lowest_low) / (highest_high - lowest_low);
        }
    }

    return stochastic;
}

// Function to compute Resistance Level
pybind11::array_t<float> computeResistance(const pybind11::array_t<float> &high_prices, int lookback) {
    if (lookback <= 0) {
        throw std::invalid_argument("Lookback must be a positive integer.");
    }

    pybind11::buffer_info buf = high_prices.request();

    if (buf.ndim != 1) {
        throw std::runtime_error("Input data should be a 1D array.");
    }

    const float *high_ptr = static_cast<const float *>(buf.ptr);
    size_t size = buf.size;

    pybind11::array_t<float> resistance(size);
    auto resistance_buf = resistance.request();
    float *resistance_ptr = static_cast<float *>(resistance_buf.ptr);

    for (size_t i = 0; i < size; ++i) {
        if (i < lookback - 1) {
            resistance_ptr[i] = 0.0f;
        } else {
            float highest_high = high_ptr[i];
            for (int j = 0; j < lookback; ++j) {
                highest_high = std::max(highest_high, high_ptr[i - j]);
            }
            resistance_ptr[i] = highest_high;
        }
    }

    return resistance;
}

// Function to compute Support Level
pybind11::array_t<float> computeSupport(const pybind11::array_t<float> &low_prices, int lookback) {
    if (lookback <= 0) {
        throw std::invalid_argument("Lookback must be a positive integer.");
    }

    pybind11::buffer_info buf = low_prices.request();

    if (buf.ndim != 1) {
        throw std::runtime_error("Input data should be a 1D array.");
    }

    const float *low_ptr = static_cast<const float *>(buf.ptr);
    size_t size = buf.size;

    pybind11::array_t<float> support(size);
    auto support_buf = support.request();
    float *support_ptr = static_cast<float *>(support_buf.ptr);

    for (size_t i = 0; i < size; ++i) {
        if (i < lookback - 1) {
            support_ptr[i] = 0.0f;
        } else {
            float lowest_low = low_ptr[i];
            for (int j = 0; j < lookback; ++j) {
                lowest_low = std::min(lowest_low, low_ptr[i - j]);
            }
            support_ptr[i] = lowest_low;
        }
    }

    return support;
}

pybind11::array_t<float> computeSMA(const pybind11::array_t<float> &data_array, int lookback) {
    if (lookback <= 0) {
        throw std::invalid_argument("Lookback must be a positive integer.");
    }

    pybind11::buffer_info buf = data_array.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input data should be a 1D array.");
    }

    const float *data_ptr = static_cast<const float *>(buf.ptr);
    size_t size = buf.size;

    pybind11::array_t<float> sma(size);
    auto sma_buf = sma.request();
    float *sma_ptr = static_cast<float *>(sma_buf.ptr);

    for (size_t i = 0; i < size; ++i) {
        if (i < lookback - 1) {
            sma_ptr[i] = 0.0f;
        } else {
            float sum = 0.0f;
            for (int j = 0; j < lookback; ++j) {
                sum += data_ptr[i - j];
            }
            sma_ptr[i] = sum / lookback;
        }
    }

    return sma;
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>, pybind11::array_t<float>>
computeBollingerBands(const pybind11::array_t<float> &data_array, int lookback, float num_stddev) {
    if (lookback <= 0 || num_stddev <= 0.0f) {
        throw std::invalid_argument("Lookback and num_stddev must be positive.");
    }

    pybind11::buffer_info buf = data_array.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input data should be a 1D array.");
    }

    const float *data_ptr = static_cast<const float *>(buf.ptr);
    size_t size = buf.size;

    pybind11::array_t<float> sma(size);
    pybind11::array_t<float> upper_band(size);
    pybind11::array_t<float> lower_band(size);

    auto sma_buf = sma.request();
    auto upper_band_buf = upper_band.request();
    auto lower_band_buf = lower_band.request();

    float *sma_ptr = static_cast<float *>(sma_buf.ptr);
    float *upper_band_ptr = static_cast<float *>(upper_band_buf.ptr);
    float *lower_band_ptr = static_cast<float *>(lower_band_buf.ptr);

    for (size_t i = 0; i < size; ++i) {
        if (i < lookback - 1) {
            sma_ptr[i] = 0.0f;
            upper_band_ptr[i] = 0.0f;
            lower_band_ptr[i] = 0.0f;
        } else {
            float sum = 0.0f;
            for (int j = 0; j < lookback; ++j) {
                sum += data_ptr[i - j];
            }
            float mean = sum / lookback;
            sma_ptr[i] = mean;

            float variance_sum = 0.0f;
            for (int j = 0; j < lookback; ++j) {
                variance_sum += std::pow(data_ptr[i - j] - mean, 2);
            }
            float stddev = std::sqrt(variance_sum / lookback);
            upper_band_ptr[i] = mean + (num_stddev * stddev);
            lower_band_ptr[i] = mean - (num_stddev * stddev);
        }
    }

    return std::make_tuple(sma, upper_band, lower_band);
}

pybind11::array_t<float> computeRSI(const pybind11::array_t<float> &data_array, int lookback) {
    if (lookback <= 0) {
        throw std::invalid_argument("Lookback must be a positive integer.");
    }

    pybind11::buffer_info buf = data_array.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input data should be a 1D array.");
    }

    const float *data_ptr = static_cast<const float *>(buf.ptr);
    size_t size = buf.size;

    pybind11::array_t<float> rsi(size);
    auto rsi_buf = rsi.request();
    float *rsi_ptr = static_cast<float *>(rsi_buf.ptr);

    float gain = 0.0f, loss = 0.0f;
    for (size_t i = 1; i < lookback; ++i) {
        float change = data_ptr[i] - data_ptr[i - 1];
        if (change > 0) {
            gain += change;
        } else {
            loss -= change;
        }
    }

    gain /= lookback;
    loss /= lookback;

    rsi_ptr[lookback - 1] = 100.0f - (100.0f / (1.0f + (gain / loss)));

    for (size_t i = lookback; i < size; ++i) {
        float change = data_ptr[i] - data_ptr[i - 1];
        if (change > 0) {
            gain = (gain * (lookback - 1) + change) / lookback;
            loss = (loss * (lookback - 1)) / lookback;
        } else {
            gain = (gain * (lookback - 1)) / lookback;
            loss = (loss * (lookback - 1) - change) / lookback;
        }

        rsi_ptr[i] = 100.0f - (100.0f / (1.0f + (gain / loss)));
    }

    return rsi;
}

pybind11::array_t<float> computeMACD(const pybind11::array_t<float> &data_array, int fast_period, int slow_period, int signal_period) {
    auto fast_ema = computeEMA(data_array, fast_period);
    auto slow_ema = computeEMA(data_array, slow_period);

    pybind11::buffer_info fast_buf = fast_ema.request();
    pybind11::buffer_info slow_buf = slow_ema.request();

    const float *fast_ptr = static_cast<const float *>(fast_buf.ptr);
    const float *slow_ptr = static_cast<const float *>(slow_buf.ptr);
    size_t size = fast_buf.size;

    pybind11::array_t<float> macd(size);
    auto macd_buf = macd.request();
    float *macd_ptr = static_cast<float *>(macd_buf.ptr);

    for (size_t i = 0; i < size; ++i) {
        macd_ptr[i] = fast_ptr[i] - slow_ptr[i];
    }

    auto signal = computeEMA(macd, signal_period);

    return signal;
}

pybind11::array_t<int> computeMovingAverageCrossover(const pybind11::array_t<float> &data_array, int short_lookback, int long_lookback) {
    auto short_sma = computeSMA(data_array, short_lookback);
    auto long_sma = computeSMA(data_array, long_lookback);

    pybind11::buffer_info short_buf = short_sma.request();
    pybind11::buffer_info long_buf = long_sma.request();

    const float *short_ptr = static_cast<const float *>(short_buf.ptr);
    const float *long_ptr = static_cast<const float *>(long_buf.ptr);
    size_t size = short_buf.size;

    pybind11::array_t<int> signals(size);
    auto signals_buf = signals.request();
    int *signals_ptr = static_cast<int *>(signals_buf.ptr);

    for (size_t i = 0; i < size; ++i) {
        if (short_ptr[i] > long_ptr[i]) {
            signals_ptr[i] = 1; // Buy signal
        } else if (short_ptr[i] < long_ptr[i]) {
            signals_ptr[i] = -1; // Sell signal
        } else {
            signals_ptr[i] = 0; // Neutral signal
        }
    }

    return signals;
}

PYBIND11_MODULE(traderlib, m) {
    m.doc() = "Module for computing technical analysis indicators";

    m.def("computeEMA", &computeEMA, "Compute Exponential Moving Average (EMA)",
          pybind11::arg("data_array"), pybind11::arg("lookback"));

    m.def("computeStochasticOscillator", &computeStochasticOscillator, "Compute Stochastic Oscillator",
          pybind11::arg("close_prices"), pybind11::arg("low_prices"), pybind11::arg("high_prices"), pybind11::arg("lookback"));

    m.def("computeResistance", &computeResistance, "Compute Resistance Level",
          pybind11::arg("high_prices"), pybind11::arg("lookback"));

    m.def("computeSupport", &computeSupport, "Compute Support Level",
          pybind11::arg("low_prices"), pybind11::arg("lookback"));

    m.def("computeSMA", &computeSMA, "Compute Simple Moving Average (SMA)",
          pybind11::arg("data_array"), pybind11::arg("lookback"));

    m.def("computeBollingerBands", &computeBollingerBands, "Compute Bollinger Bands",
          pybind11::arg("data_array"), pybind11::arg("lookback"), pybind11::arg("num_stddev"));

    m.def("computeRSI", &computeRSI, "Compute Relative Strength Index (RSI)",
          pybind11::arg("data_array"), pybind11::arg("lookback"));

    m.def("computeMACD", &computeMACD, "Compute Moving Average Convergence Divergence (MACD)",
          pybind11::arg("data_array"), pybind11::arg("fast_period"), pybind11::arg("slow_period"), pybind11::arg("signal_period"));

    m.def("computeMovingAverageCrossover", &computeMovingAverageCrossover, "Compute Moving Average Crossover signals",
          pybind11::arg("data_array"), pybind11::arg("short_lookback"), pybind11::arg("long_lookback"));
}