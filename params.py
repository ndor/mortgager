'''
http://www.yorku.ca/amarshal/mortgage.htm
https://www.moti.org.il/intrests
'''

SEED = 108

# resolution params:
CALC_INCREMENTS = 30
dt_m = 12
dt_y = 5
MAX_DURATION = 30 * dt_m
MIN_DURATION = dt_y * dt_m
BANK_BASE = {'spitzer': {'fixed': {}, 'madad': {}, 'prime': {}}, 'equal': {'fixed': {}, 'madad': {}, 'prime': {}}}
DURATIONS = list(range(MIN_DURATION, MAX_DURATION + MIN_DURATION, MIN_DURATION))  # duration in step, by years strides

# financial constraint params:
MAX_UNFIXED_PORTON = 0.667
MIN_UNFIXED_PORTON = 0.1
MIN_FIXED_PORTION = 0.333
MAX_FUNDING_RATE_FOR_FIRST_APPARTMENT = 0.75
MAX_FUNDING_RATE_FOR_NON_FIRST_APPARTMENT = 0.5
ALLOWABLE_MONTHLY_FRACTION = 0.333

# rate array params:
FIXED_VALUE = 0.025
MADAD_INITIAL_VALUE = 0.025
PRIME_INITIAL_VALUE = 0.025
INITIAL_STEADY_PERIOD_MONTHS_DURATION = 12
STEADY_RAMP_MONTHS_DURATION = 84
LONG_RANGE_CENTERLINE = 0.03
BANKS_MARGINE = 0.015
PRIME_ADDED_YEARLY_RATE = -0.0064

