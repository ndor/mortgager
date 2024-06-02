import numpy as np
from scipy.interpolate import interp1d
import numpy_financial_functions as npff
from params import *


def get_madad(initial_value=MADAD_INITIAL_VALUE,
              initial_steady_period_months_duration=INITIAL_STEADY_PERIOD_MONTHS_DURATION,
              steady_ramp_months_duration=STEADY_RAMP_MONTHS_DURATION,
              long_range_centerline=LONG_RANGE_CENTERLINE):

    madad_0 = initial_value * np.ones(initial_steady_period_months_duration, dtype='float16')
    madad_1 = np.polyval([(long_range_centerline - initial_value) / steady_ramp_months_duration, madad_0[-1]],
                         np.arange(steady_ramp_months_duration))
    madad_2 = madad_1[-1] * np.ones(MAX_DURATION)
    madad = np.concatenate([madad_0, madad_1, madad_2]).astype('float16')
    return madad


def get_prime(initial_value=PRIME_INITIAL_VALUE,
              initial_steady_period_months_duration=INITIAL_STEADY_PERIOD_MONTHS_DURATION,
              steady_ramp_months_duration=STEADY_RAMP_MONTHS_DURATION,
              long_range_centerline=LONG_RANGE_CENTERLINE,
              banks_margine=BANKS_MARGINE,
              wavewlwngthy=10,
              amplitude=0.015):
    try:
        initial_value = requests.get('https://Boi.org.il/PublicApi/GetInterest').json()['currentInterest'] / 100
    except:
        pass
    prime_0 = initial_value * np.ones(initial_steady_period_months_duration, dtype='float16') + banks_margine
    prime_1 = np.polyval([(long_range_centerline - prime_0[-1]) / steady_ramp_months_duration, prime_0[-1]],
                         np.arange(steady_ramp_months_duration))
    prime_2 = np.sin(2 * np.pi * np.arange(0, 30, 1 / 12) / wavewlwngthy) * amplitude + prime_1[-1]
    prime = np.concatenate([prime_0, prime_1, prime_2]).astype('float16')
    return prime


monthly_changing_yearly_MADAD = get_madad()
monthly_changing_yearly_PRIME = get_prime()

def yearly_rate_to_monthly(yearly_rate) -> np.ndarray:
    return yearly_rate / 12

def nominal_interest_rate_to_effective(yearly_rate: np.ndarray) -> np.ndarray:
    return (1 + yearly_rate_to_monthly(yearly_rate)) ** 12 - 1

def get_spitzer_amortization(monthly_rate_array: np.ndarray, duration: int, principal: int) -> dict:
    per = np.arange(duration) + 1 # periods (months)
    ipmt = - npff.ipmt(monthly_rate_array[:duration], per, duration, principal) # interest portion of a payment
    ppmt = (principal / duration) * np.ones_like(per)  # payment against loan principal
    pmt = ((ipmt.sum() + principal) / duration) * np.ones_like(per) # monthly payment against loan principal plus interest
    return {'pmt': pmt.astype('float16'), 'ipmt': ipmt.astype('float16'), 'ppmt': ppmt.astype('float16')}

def get_equal_amortization(monthly_rate_array, duration, principal) -> dict:
    per = np.arange(duration) + 1 # periods (months)
    ppmt = principal / duration * np.ones_like(per) # payment against loan principal
    ipmt = (principal - ppmt.cumsum()) * monthly_rate_array[:duration] # interest portion of a payment
    pmt = ipmt + ppmt # monthly payment against loan principal plus interest
    return {'pmt': pmt.astype('float16'), 'ipmt': ipmt.astype('float16'), 'ppmt': ppmt.astype('float16')}

fixed_yearly_risk_rate = lambda r: interp1d([0., .45, .6, .7, .75],
                                            [.028, .0285, .03, .0315, .0315], kind='cubic')(r)
madad_added_risk_yearly_rate = lambda r: interp1d([0., .45, .6, .7, .75],
                                                  [.018, .019, .0205, .0225, .0225], kind='cubic')(r)

def risk_rating(funding_rate: float, is_married_couple=False) -> dict:
    un_married_rate = 1. if is_married_couple else 1.1
    _fixed_yearly_risk_rate = 1 + fixed_yearly_risk_rate(funding_rate) * un_married_rate
    _madad_added_risk_yearly_rate = 1 + madad_added_risk_yearly_rate(funding_rate) * un_married_rate
    return {'prime_yearly_added_risk_rate': 1.0001 * un_married_rate,
            'fixed_yearly_added_risk_rate': _fixed_yearly_risk_rate,
            'madad_yearly_added_risk_yearly_rate': _madad_added_risk_yearly_rate}

def update_yearly_to_monthly_rates_with_risk(funding_rate: float, is_married_couple=False) -> dict:
    risk = risk_rating(funding_rate, is_married_couple=is_married_couple)
    fixed_rate = np.ones_like(monthly_changing_yearly_MADAD) * FIXED_VALUE
    fixed_rate = fixed_rate * risk['fixed_yearly_added_risk_rate']
    madad_rate = monthly_changing_yearly_MADAD * risk['madad_yearly_added_risk_yearly_rate']
    prime_rate = (monthly_changing_yearly_PRIME + PRIME_ADDED_YEARLY_RATE) * risk['fixed_yearly_added_risk_rate']
    return {'fixed_rate': yearly_rate_to_monthly(fixed_rate).astype('float16'),
            'madad_rate': yearly_rate_to_monthly(madad_rate).astype('float16'),
            'prime_rate': yearly_rate_to_monthly(prime_rate).astype('float16')}



if __name__ == '__main__':

    '''
    fv(rate, nper, pmt, pv[, when])	Compute the future value.
    pv(rate, nper, pmt[, fv, when])	Compute the present value.
    npv(rate, values)	Returns the NPV (Net Present Value) of a cash flow series.
    pmt(rate, nper, pv[, fv, when])	Compute the payment against loan principal plus interest.
    ppmt(rate, per, nper, pv[, fv, when])	Compute the payment against loan principal.
    ipmt(rate, per, nper, pv[, fv, when])	Compute the interest portion of a payment.
    irr(values)	Return the Internal Rate of Return (IRR).
    mirr(values, finance_rate, reinvest_rate)	Modified internal rate of return.
    nper(rate, pmt, pv[, fv, when])	Compute the number of periodic payments.
    rate(nper, pmt, pv, fv[, when, guess, tol, …])	Compute the rate of interest per period.
    '''

    asset_cost = 2150000
    capital = 950000

    import matplotlib.pyplot as plt
    import pandas as pd
    # plt.plot(monthly_changing_yearly_PRIME)
    # plt.plot(monthly_changing_yearly_MADAD)
    # plt.show()

    # monthly_rate = yearly_rate_to_monthly(monthly_changing_yearly_MADAD)
    # # monthly_rate = yearly_rate_to_monthly(0.02 * np.ones(30 * 12))
    # duration = 30 * 12
    # principal = asset_cost - capital
    # m = get_spitzer_amortization(monthly_rate, duration, principal)
    # pd.DataFrame.from_dict(m).plot()
    # n = get_equal_amortization(monthly_rate, duration, principal)
    # pd.DataFrame.from_dict(n).plot()
    # plt.show()

    loan_principal = asset_cost - capital
    funding_rate = loan_principal / asset_cost
    d = update_yearly_to_monthly_rates_with_risk(funding_rate, False, 0.035)
    # pd.DataFrame.from_dict(d).plot()
    # plt.show()

