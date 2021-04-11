# load and clean-up the power usage dataset
from numpy import nan
from numpy import isnan
from pandas import read_csv
import os


# fill missing values with a value at the same time one day ago
def fill_missing(values):
    one_day = 60 * 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if isnan(values[row, col]):
                values[row, col] = values[row - one_day, col]

# load all data
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'household_power_consumption.txt')
dataset = read_csv(filename, sep=';', header=0, low_memory=False,
                    infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
# mark all missing values
dataset.replace('?', nan, inplace=True) # make dataset numeric
dataset = dataset.astype('float32')
# fill missing fill_missing(dataset.values)
# add a column for for the remainder of sub metering
values = dataset.values
dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] +
    values[:,6])
# save updated dataset
dataset.to_csv('household_power_consumption.csv')

# resample minute data to total for each day for the power usage dataset
from pandas import read_csv
# load the new file
# resample data to daily
daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()
# summarize
print(daily_data.shape)
# print(daily_data.head())
# save
daily_data.to_csv('household_power_consumption_days.csv')