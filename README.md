# Individual Household Electric Power Consumption

## Data

The data is taken from the UC Irvine Machine Learning Repository: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The electricity consumption from a household in France are observed for about 4 years, from December 2006 to November 2010. Measurements are made every minute.

The 7 timeseries variables are:

- `Global_active_power`: household global minute-averaged active power (in kilowatt)
- `Global_reactive_power`: household global minute-averaged reactive power (in kilowatt)
- `Voltage`: minute-averaged voltage (in volt)
- `Global_intensity`: household global minute-averaged current intensity (in  ampere)
- `Sub_metering_1`: energy sub-metering No. 1 (in watt-hour of active energy) corresponding to the kitchen
- `Sub_metering_2`: energy sub-metering No. 2 (in watt-hour of active energy) corresponding to the laundry room
- `Sub_metering_3`: energy sub-metering No. 3 (in watt-hour of active energy) corresponding to an electric water-heater and an air-conditioner

To reduce the data size, measurements are aggregated by grouping by hour and taking the mean, so I end up with hourly measurements.

## Problem statement

In a alternative current circuit, the active power and the reactive power are respectively the real part and the imaginary part of the *complex power*. The *apparent power* is defined as the modulus of the complex power.

I aim at predicting the apparent energy consumed in the next 24 hours given the data from the last 24 hours.

See: https://en.wikipedia.org/wiki/AC_power

## Model setup

At time $t$ (hourly time index), let $y_t$ be the apparent power and $x_t$ be the 7-dimensional measured vector of variables listed above. I augment $x_t$ with 3 time variables which I suppose carry information about electricity consumption: the current hour (from 0 to 23), weekday (from 0 for Monday to 6 for Sunday), day of year (from 0 for January 1st to 365 or 366 for December 31st). Let $x'_t$ be the augmented 10-dimensional vector.

Then let $Y_t = y_{t+1} + \cdots + y_{t+24}$ be the apparent energy (in kilojoule) of the next 24 hours and $X_t = (x'_{t-23}, \dots, x'_t)$ be the observations from the last 24 hours.

The model $F$ to be learned writes $Y_t = F(X_t)$, where $Y_t$ is a scalar and $X_t$ is a $24 \times 10$ matrix.

## Results

## How to use
