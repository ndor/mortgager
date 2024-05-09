'''
http://www.yorku.ca/amarshal/mortgage.htm
https://www.moti.org.il/intrests
'''

SEED = 108
CALC_INCREMENTS = 30
dt_m = 12
dt_y = 5
MAX_PRIME_PORTON = 0.667
MINIMAL_FIXED_PORTION = 0.333
MAX_FUNDING_RATE_FOR_FIRST_APPARTMENT = 0.75
MAX_FUNDING_RATE_FOR_NON_FIRST_APPARTMENT = 0.5
MAX_DURATION = 30 * dt_m
MIN_DURATION = 5 * dt_m
ALLOWABLE_MONTHLY_FRACTION = 0.333
BANK_BASE = {'shpitzer': {'fixed': {}, 'madad': {}, 'prime': {}}, 'equal': {'fixed': {}, 'madad': {}, 'prime': {}}}
DURATIONS = list(range(MIN_DURATION, MAX_DURATION + dt_m * dt_y, dt_m * dt_y))  # duration in step, by years strides

