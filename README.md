# Individual Household Electric Power Consumption

## Data

The data is taken from the UC Irvine Machine Learning Repository: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The electricity consumption from a household in France is observed for ~4 years, from December 2006 to November 2010. Measurements are made every minute.

Seven timeseries are available:

- `Global_active_power`: household global minute-averaged active power (in kilowatt)
- `Global_reactive_power`: household global minute-averaged reactive power (in kilowatt)
- `Voltage`: minute-averaged voltage (in volt)
- `Global_intensity`: household global minute-averaged current intensity (in  ampere)
- `Sub_metering_1`: energy sub-metering (in watt-hour of active energy) corresponding to the kitchen
- `Sub_metering_2`: energy sub-metering (in watt-hour of active energy) corresponding to the laundry room
- `Sub_metering_3`: energy sub-metering (in watt-hour of active energy) corresponding to an electric water-heater and an air-conditioner.

To reduce the data size, measurements are aggregated by grouping by hour and taking the average, so I end up with hourly measurements.

## Problem statement

In a alternative current circuit, the active power and the reactive power are respectively the real part and the imaginary part of the ***complex power***. The ***apparent power*** is defined as the modulus of the complex power.

See: https://en.wikipedia.org/wiki/AC_power

I aim at forecasting, at each time step, the apparent energy consumed in the next 24 hours given the data from the previous 24 hours.

## Model setup

At time $t$ (hourly time index), let $y_t$ be the apparent power and $x_t$ be the 7-dimensional measured vector of variables listed above. I augment $x_t$ with 3 time variables which supposedly carry information about electricity consumption:

- current hour (from 0 to 23)
- current weekday (from 0 for Monday to 6 for Sunday)
- current day of year (from 0 for January 1st to 365 or 366 for December 31st).

Let $x'_t$ be the augmented 10-dimensional vector.

Then let $Y_t = y_{t+1} + \cdots + y_{t+24}$ be the apparent energy (in kilojoule) of the next 24 hours. Let $X_t = (x'_{t-23}, \dots, x'_t)$ be the stack of observations from the last 24 hours (each row vector is stacked vertically).

The model $F$ to be learned writes $Y_t = F(X_t)$, where $Y_t$ is a scalar and $X_t$ is a $24 \times 10$ matrix.

The chosen model design for $F$ is a neural network with upstream recurrent layers (LSTM) and downstream fully connected layers. The loss function is the mean squared error.

## Results

The data is split as follows: 50% (Dec 2006 to Nov 2008) for training, 25% (Dec 2008 to Nov 2009) for validation (loss monitoring to stop training to avoid overfitting), 25% (Dec 2009 to Nov 2010) for test.

<p align="center">
  <img src="https://github.com/paulbuiqg/houselec/blob/main/viz/training_history.png" />
</p>

The trained model prediction has a [normalized mean absolute error](https://agrimabahl.medium.com/mape-v-s-mae-v-s-rmse-3e358fd58f65) of ~0.20 on the test set. Results may (marginally) vary due to random shuffling of the training data batches and random initialization of the neural network parameters.

## How to use

- Go to the repo root directory
- Install required libraries: `pip install -r requirements.txt`
- Run: `python3 src/main.py`
- For unit testing, run: `pytest`
