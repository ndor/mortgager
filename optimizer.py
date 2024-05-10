import copy
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, brute, shgo, minimize, LinearConstraint, NonlinearConstraint

try:
    import requests
except:
    pass
    # import pyodide_js
    # await pyodide_js.loadPackage('requests')

import numpy_financial_functions as npff
from params import *

import warnings
warnings.filterwarnings('ignore')


def get_madad(initial_value=0.025,
              initial_steady_period_months_duration=12,
              steady_ramp_months_duration=84,
              long_range_centerline=0.03):

    madad_0 = initial_value * np.ones(initial_steady_period_months_duration, dtype='float16')
    madad_1 = np.polyval([(long_range_centerline - initial_value) / steady_ramp_months_duration, madad_0[-1]],
                         np.arange(steady_ramp_months_duration))
    madad_2 = madad_1[-1] * np.ones(MAX_DURATION)
    madad = np.concatenate([madad_0, madad_1, madad_2]).astype('float16')
    return madad


def get_prime(initial_value=0.045,
              initial_steady_period_months_duration=12,
              steady_ramp_months_duration=84,
              long_range_centerline=0.03,
              banks_margine=0.015,
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
    # import matplotlib.pyplot as plt
    # plt.plot(prime)
    # plt.show()
    return prime


MADAD = get_madad()
PRIME = get_prime()


def yearly_rate_to_monthly(yearly_rate):
    return yearly_rate / 12


def nominal_interest_rate_to_effective(yearly_rate):
    return (1 + yearly_rate_to_monthly(yearly_rate)) ** 12 - 1


def get_shpitzer_amortization(monthly_rate, duration, principal, return_dict=False, return_lists=False):
    try:
        len(monthly_rate)
    except TypeError:
        monthly_rate = monthly_rate * np.ones(duration, dtype='float16')
    pmt = npff.pmt(monthly_rate[:duration], duration, principal, fv=0, when=0)  # monthly payment against loan principal plus interest
    per = np.arange(duration) + 1  # periods (months)
    ipmt = npff.ipmt(monthly_rate[:duration], per, duration, principal)  # interest portion of a payment
    ppmt = npff.ppmt(monthly_rate[:duration], per, duration, principal)  # payment against loan principal
    if return_lists:
        d = {
            'pmt': pmt.tolist(),
            'ipmt': ipmt.tolist(),
            'ppmt': ppmt.tolist()
        }
    else:
        d = {
            'pmt': pmt,
            'ipmt': ipmt,
            'ppmt': ppmt
        }
    if return_dict:
        return d
    return pd.DataFrame.from_dict(d)


def get_equal_amortization(monthly_rate, duration, principal, return_dict=False, return_lists=False):
    try:
        len(monthly_rate)
    except TypeError:
        monthly_rate = monthly_rate * np.ones(duration)
    per = np.arange(duration) + 1  # periods (months)
    ppmt = - principal / duration * np.ones_like(per)  # payment against loan principal
    ipmt = - (principal + ppmt.cumsum()) * monthly_rate[:duration]  # interest portion of a payment
    pmt = ipmt + ppmt  # monthly payment against loan principal plus interest
    if return_lists:
        d = {
            'pmt': pmt.tolist(),
            'ipmt': ipmt.tolist(),
            'ppmt': ppmt.tolist()
        }
    else:
        d = {
            'pmt': pmt,
            'ipmt': ipmt,
            'ppmt': ppmt
        }
    if return_dict:
        return d
    return pd.DataFrame.from_dict(d)


def build_base_amortization_bank_dictionary(fixed_monthly_rate,
                                            madad_monthly_rate,
                                            prime_monthly_rate,
                                            shpitzer=True,
                                            equal=True,
                                            return_dict=False,
                                            return_lists=False):
    bank = {}
    if shpitzer:
        bank['shpitzer'] = {'fixed': {}, 'madad': {}, 'prime': {}}
    if equal:
        bank['equal'] = {'fixed': {}, 'madad': {}, 'prime': {}}

    for title, monthly_rate in [('fixed', fixed_monthly_rate),
                                ('madad', madad_monthly_rate),
                                ('prime', prime_monthly_rate)]:
        for duration in DURATIONS:
            if shpitzer:
                bank['shpitzer'][title][duration] = get_shpitzer_amortization(monthly_rate,
                                                                              duration,
                                                                              1, #.00278,
                                                                              return_dict=return_dict,
                                                                              return_lists=return_lists)
            if equal:
                bank['equal'][title][duration] = get_equal_amortization(monthly_rate,
                                                                        duration,
                                                                        1, #.00278,
                                                                        return_dict=return_dict,
                                                                        return_lists=return_lists)
    return bank


def convert_bank_to_singular_values(bank):
    _bank = {}
    for amortization_type in bank.keys():
        _bank[amortization_type] = {}
        for rate_type in bank[amortization_type].keys():
            _bank[amortization_type][rate_type] = {}
            for duration in bank[amortization_type][rate_type].keys():
                _bank[amortization_type][rate_type][duration] = {}
                _bank[amortization_type][rate_type][duration]['pmt'] = \
                    bank[amortization_type][rate_type][duration]['pmt'][0]
                _bank[amortization_type][rate_type][duration]['ppmt'] = \
                    bank[amortization_type][rate_type][duration]['ppmt'].sum()
                _bank[amortization_type][rate_type][duration]['ipmt'] = \
                    bank[amortization_type][rate_type][duration]['ipmt'].sum()
    return _bank


def principize_base_bank(principals, amortization_type, max_monthly_payment, bank):
    _bank = {amortization_type: {}}
    for rate_type in principals.keys():
        _bank[amortization_type][rate_type] = {}
        for duration in bank[amortization_type][rate_type].keys():
            if bank[amortization_type][rate_type][duration]['pmt'] * principals[rate_type] > max_monthly_payment:
                continue
            _bank[amortization_type][rate_type][duration] = {}
            for k in bank[amortization_type][rate_type][duration].keys():
                _bank[amortization_type][rate_type][duration][k] = \
                    bank[amortization_type][rate_type][duration][k] * principals[rate_type]
    return _bank


def amortized_bundle_filter_by_max_monthly_payment(_bank, amortization_type, max_monthly_payment):
    bundles = []
    for a_duration in _bank[amortization_type]['fixed'].keys():
        a_max_pmt = _bank[amortization_type]['fixed'][a_duration]['pmt']
        if -max_monthly_payment > a_max_pmt:
            continue
        for b_duration in _bank[amortization_type]['madad'].keys():
            b_max_pmt = _bank[amortization_type]['madad'][b_duration]['pmt'] + a_max_pmt
            if -max_monthly_payment > b_max_pmt:
                continue
            for c_duration in _bank[amortization_type]['prime'].keys():
                c_max_pmt = _bank[amortization_type]['prime'][c_duration]['pmt'] + b_max_pmt
                if -max_monthly_payment > c_max_pmt:
                    continue
                bundles.append((
                    (amortization_type, 'fixed', a_duration),
                    (amortization_type, 'madad', b_duration),
                    (amortization_type, 'prime', c_duration)
                ))
    return bundles


def max_monthly_payment_for_bundle(bundle, _bank):
    return sum(map(lambda b: abs(_bank[b[0]][b[1]][b[2]]['pmt']), bundle))


def total_ipmt_for_bundle(bundle, _bank):
    return sum(map(lambda b: np.abs(_bank[b[0]][b[1]][b[2]]['ipmt']), bundle))


def optimize_bundles_filter(_bank, bundles, for_optimization=False):
    opt_bundle = ()
    opt_bundle_tot = 1e12
    for bundle in bundles:
        bundle_tot = total_ipmt_for_bundle(bundle, _bank)
        if bundle_tot < opt_bundle_tot:
            opt_bundle_tot = bundle_tot
            opt_bundle = bundle

    if for_optimization:
        return opt_bundle_tot

    opt = copy.deepcopy(BANK_BASE)
    for b in opt_bundle:
        if b[2] == 0:
            continue
        opt[b[0]][b[1]][b[2]] = _bank[b[0]][b[1]][b[2]]
    return opt


def X2D_X3D(X, principal):
    return np.append(X, [principal - sum(X)])


def X_to_dict(X):
    return {'fixed': X[0], 'madad': X[1], 'prime': X[2]}


def amortization_target_function(X, amortization_type, max_monthly_payment, bank, for_optimization=True):
    _bank = principize_base_bank(X_to_dict(X), amortization_type, max_monthly_payment, bank)
    bundles = amortized_bundle_filter_by_max_monthly_payment(_bank, amortization_type, max_monthly_payment)
    return optimize_bundles_filter(_bank, bundles, for_optimization=for_optimization)


def optimize_amortization(principal,
                          amortization_type,
                          max_monthly_payment,
                          bank,
                          use_max_prime=False,
                          increments=CALC_INCREMENTS,
                          max_prime_content=MAX_PRIME_PORTON):
    fun = lambda X: amortization_target_function(X,
                                                  amortization_type,
                                                  max_monthly_payment,
                                                  bank,
                                                  for_optimization=True)
    result = None
    if use_max_prime:
        x3 = principal * max_prime_content
        slack = principal - x3
        min_f_val = np.inf
        for x1 in np.linspace(MINIMAL_FIXED_PORTION * principal, slack, increments):  # limit fixed
            x2 = slack - x1
            f_val = fun((x1, x2, x3))
            if np.isinf(f_val):
                continue
            if f_val < min_f_val:
                min_f_val = f_val
                result = (x1, x2, x3)
        if result:
            best_bundle = amortization_target_function(X2D_X3D(result, principal),
                                                       amortization_type,
                                                       max_monthly_payment,
                                                       bank,
                                                       for_optimization=False)
    else:
        min_f_val = np.inf
        for x1 in np.linspace(MINIMAL_FIXED_PORTION * principal, principal, increments):  # limit fixed
            for x2 in np.linspace(0, principal - x1, increments):
                if x1 + x2 <= principal:
                    if principal - (x1 + x2) <= max_prime_content * principal:  # limit prime
                        f_val = func(X2D_X3D((x1, x2), principal))
                        if f_val < min_f_val:
                            min_f_val = f_val
                            result = (x1, x2)
        if result:
            best_bundle = amortization_target_function(X2D_X3D(result, principal),
                                                       amortization_type,
                                                       max_monthly_payment,
                                                       bank,
                                                       for_optimization=False)

    if result:
        return best_bundle, min_f_val
    else:
        return None, np.inf


def sleek_optimize_amortization(principal,
                                amortization_type,
                                max_monthly_payment,
                                bank,
                                use_max_prime=False,
                                max_prime_content=MAX_PRIME_PORTON):
    if use_max_prime:
        func = lambda X: amortization_target_function([X,
                                                       principal * (1 - max_prime_content) - X,
                                                       principal * max_prime_content],
                                                      amortization_type,
                                                      max_monthly_payment,
                                                      bank,
                                                      for_optimization=True)

        bounds = [(MINIMAL_FIXED_PORTION * principal, principal * (1 - max_prime_content))]
        # constraints = [
        #     NonlinearConstraint(lambda x: x.sum(), principal, principal),
        # ]
        result = differential_evolution(func,
                                        bounds,
                                        # constraints=constraints,
                                        disp=False,
                                        # maxiter=10,
                                        atol=1e3,
                                        tol=0.0001,
                                        popsize=9,
                                        seed=SEED,
                                        workers=1)
        best_bundle = amortization_target_function([result.x,
                                                    principal * (1 - max_prime_content) - result.x,
                                                    principal * max_prime_content],
                                                   amortization_type,
                                                   max_monthly_payment,
                                                   bank,
                                                   for_optimization=False)
    else:
        func = lambda X: amortization_target_function(X2D_X3D(X, principal),
                                                      amortization_type,
                                                      max_monthly_payment,
                                                      bank,
                                                      for_optimization=True)

        bounds = [(MINIMAL_FIXED_PORTION * principal, principal), (0., loan_principal)]
        # constraints = [
        #     NonlinearConstraint(lambda x: principal - x.sum(), 0, max_prime_content * principal),
        #     # NonlinearConstraint(lambda x: x.sum(), principal, principal),
        # ]

        result = differential_evolution(func,
                                        bounds,
                                        # constraints=constraints,
                                        disp=False,
                                        # maxiter=10,
                                        atol=1e3,
                                        tol=0.0001,
                                        popsize=9,
                                        seed=SEED,
                                        workers=1)
        best_bundle = amortization_target_function(X2D_X3D(result.x, principal),
                                                   amortization_type,
                                                   max_monthly_payment,
                                                   bank,
                                                   for_optimization=False)
    return best_bundle, result.fun


def optimize_mortgage(loan_principal,
                      max_monthly_payment,
                      amortizations,
                      use_max_prime=True,
                      fixed_yearly_rate=0.035,
                      madad_added_yearly_rate=0.0001,
                      prime_added_yearly_rate=0.0001,
                      increments=CALC_INCREMENTS,
                      max_prime_content=MAX_PRIME_PORTON):
    fixed_monthly_rate = yearly_rate_to_monthly(fixed_yearly_rate)
    madad_monthly_rate = yearly_rate_to_monthly(MADAD + madad_added_yearly_rate)
    prime_monthly_rate = yearly_rate_to_monthly(PRIME + prime_added_yearly_rate)

    # rates = {'fixed': fixed_yearly_rate,
    #          'madad': MADAD[0] + madad_added_yearly_rate,
    #          'prime': PRIME[0] + prime_added_yearly_rate}

    bank = build_base_amortization_bank_dictionary(fixed_monthly_rate,
                                                   madad_monthly_rate,
                                                   prime_monthly_rate,
                                                   return_dict=False,
                                                   return_lists=False)

    bank_singular = convert_bank_to_singular_values(bank)
    if len(amortizations) == 1:
        # Shpitzer or Equal:
        best_bundle, min_f_val = optimize_amortization(loan_principal,
                                                       amortizations[0],
                                                       max_monthly_payment,
                                                       bank_singular,
                                                       use_max_prime=use_max_prime,
                                                       max_prime_content=max_prime_content)

        if amortizations[0] == 'shpitzer':
            best_bundle.pop('equal')
        else:
            best_bundle.pop('shpitzer')
    else:
        # Shpitzer & Equal:
        min_f_val = np.inf
        for shpitzer_principal in np.linspace(0, loan_principal, increments):
            equal_principal = loan_principal - shpitzer_principal
            for shpitzer_max_monthly_payment in np.linspace(0, max_monthly_payment, increments):
                equal_max_monthly_payment = max_monthly_payment - shpitzer_max_monthly_payment

                if (equal_max_monthly_payment < equal_principal / DURATIONS[-1]) or \
                        (shpitzer_max_monthly_payment < shpitzer_principal / DURATIONS[-1]):
                    continue

                if shpitzer_principal == 0:
                    best_shpitzer_bundle, shpitzer_min_f_val = copy.deepcopy(BANK_BASE), 0
                else:
                    best_shpitzer_bundle, shpitzer_min_f_val = optimize_amortization(shpitzer_principal,
                                                                                     'shpitzer',
                                                                                     shpitzer_max_monthly_payment,
                                                                                     bank_singular,
                                                                                     use_max_prime=use_max_prime,
                                                                                     max_prime_content=max_prime_content)
                    if best_shpitzer_bundle is None:
                        continue

                if equal_principal == 0:
                    best_equal_bundle, equal_min_f_val = copy.deepcopy(BANK_BASE), 0
                else:
                    best_equal_bundle, equal_min_f_val = optimize_amortization(equal_principal,
                                                                               'equal',
                                                                               equal_max_monthly_payment,
                                                                               bank_singular,
                                                                               use_max_prime=use_max_prime,
                                                                               max_prime_content=max_prime_content)
                    if best_equal_bundle is None:
                        continue
                f_val = equal_min_f_val + shpitzer_min_f_val
                if 0 < f_val < min_f_val:
                    min_f_val = f_val
                    best_bundle = {'shpitzer': best_shpitzer_bundle['shpitzer'], 'equal': best_equal_bundle['equal']}

        tot = 0
        for k1 in best_bundle.keys():
            for k2 in best_bundle[k1].keys():
                for k3 in best_bundle[k1][k2].keys():
                    tot += best_bundle[k1][k2][k3]['ppmt']
        c_tot = loan_principal / abs(tot)
        for k1 in best_bundle.keys():
            for k2 in best_bundle[k1].keys():
                for k3 in best_bundle[k1][k2].keys():
                    best_bundle[k1][k2][k3]['ppmt'] *= c_tot
    return best_bundle #, rates


def sleek_optimize_mortgage(loan_principal,
                            max_monthly_payment,
                            amortizations,
                            use_max_prime=True,
                            fixed_yearly_rate=0.035,
                            madad_added_yearly_rate=0.0001,
                            prime_added_yearly_rate=0.0001,
                            max_prime_content=MAX_PRIME_PORTON):
    fixed_monthly_rate = yearly_rate_to_monthly(fixed_yearly_rate)
    madad_monthly_rate = yearly_rate_to_monthly(MADAD + madad_added_yearly_rate)
    prime_monthly_rate = yearly_rate_to_monthly(PRIME + prime_added_yearly_rate)

    rates = {'fixed': fixed_yearly_rate,
             'madad': MADAD[0] + madad_added_yearly_rate,
             'prime': PRIME[0] + prime_added_yearly_rate}

    bank = build_base_amortization_bank_dictionary(fixed_monthly_rate,
                                                   madad_monthly_rate,
                                                   prime_monthly_rate,
                                                   return_dict=False,
                                                   return_lists=False)
    bank_singular = convert_bank_to_singular_values(bank)
    if len(amortizations) == 1:
        # Shpitzer or Equal:
        best_bundle, min_f_val = sleek_optimize_amortization(loan_principal,
                                                             amortizations[0],
                                                             max_monthly_payment,
                                                             bank_singular,
                                                             use_max_prime=use_max_prime,
                                                             max_prime_content=max_prime_content)
        if amortizations[0] == 'shpitzer':
            best_bundle.pop('equal')
        else:
            best_bundle.pop('shpitzer')
        tot = 0
        for k1 in best_bundle.keys():
            for k2 in best_bundle[k1].keys():
                for k3 in best_bundle[k1][k2].keys():
                    tot += best_bundle[k1][k2][k3]['ppmt']
        c_tot = loan_principal / abs(tot)
        for k1 in best_bundle.keys():
            for k2 in best_bundle[k1].keys():
                for k3 in best_bundle[k1][k2].keys():
                    best_bundle[k1][k2][k3]['ppmt'] *= c_tot
        return best_bundle
    else:
        # Shpitzer & Equal:
        def fun(X, for_opt=True):
            shpitzer_principal, shpitzer_max_monthly_payment = X
            equal_principal = loan_principal - shpitzer_principal
            equal_max_monthly_payment = max_monthly_payment - shpitzer_max_monthly_payment
            if shpitzer_principal == 0:
                best_shpitzer_bundle, shpitzer_min_f_val = copy.deepcopy(BANK_BASE), 0
            else:
                best_shpitzer_bundle, shpitzer_min_f_val = sleek_optimize_amortization(shpitzer_principal,
                                                                                       'shpitzer',
                                                                                       shpitzer_max_monthly_payment,
                                                                                       bank_singular,
                                                                                       use_max_prime=use_max_prime,
                                                                                       max_prime_content=max_prime_content)
            if equal_principal == 0:
                best_equal_bundle, equal_min_f_val = copy.deepcopy(BANK_BASE), 0
            else:
                best_equal_bundle, equal_min_f_val = sleek_optimize_amortization(equal_principal,
                                                                                 'equal',
                                                                                 equal_max_monthly_payment,
                                                                                 bank_singular,
                                                                                 use_max_prime=use_max_prime,
                                                                                 max_prime_content=max_prime_content)
            # f_val = equal_min_f_val + shpitzer_min_f_val
            # if 0 < f_val < min_f_val:
            #     min_f_val = f_val
            if for_opt:
                return equal_min_f_val + shpitzer_min_f_val
            best_bundle = {'shpitzer': best_shpitzer_bundle['shpitzer'], 'equal': best_equal_bundle['equal']}
            tot = 0
            for k1 in best_bundle.keys():
                for k2 in best_bundle[k1].keys():
                    for k3 in best_bundle[k1][k2].keys():
                        tot += best_bundle[k1][k2][k3]['ppmt']
            c_tot = loan_principal / abs(tot)
            for k1 in best_bundle.keys():
                for k2 in best_bundle[k1].keys():
                    for k3 in best_bundle[k1][k2].keys():
                        best_bundle[k1][k2][k3]['ppmt'] *= c_tot
            return best_bundle

        bounds = [(0, loan_principal), (0., max_monthly_payment)]
        # constraints = [
        #     NonlinearConstraint(lambda x: x.sum(), principal, principal),
        # ]
        result = differential_evolution(fun,
                                        bounds,
                                        # constraints=constraints,
                                        disp=False,
                                        # maxiter=10,
                                        atol=1e3,
                                        tol=0.0001,
                                        popsize=9,
                                        seed=SEED,
                                        workers=1)
        return fun(result.x, for_opt=False), rates



def bundle_to_df(bundle,
                 fixed_yearly_rate=0.035,
                 madad_added_yearly_rate=0.0001,
                 prime_added_yearly_rate=0.0001):
    if type(bundle) == tuple:
        bundle = bundle[0]

    d = {'amortization_type': [],
         'rate_type': [],
         'nominal_rate': [],
         'duration': [],
         'monthly_1st_payment': [],
         # 'monthly_max_payment': [],
         'principal': [],
         'principal_portion': [],
         'interest_paid': [],
         'net_paid': [],
         'returned_ratio': [],
         'effective_overall_rate': []}

    tot_principal = 0
    for amortization_type in bundle.keys():
        for rate_type in bundle[amortization_type].keys():
            for duration in bundle[amortization_type][rate_type].keys():
                tot_principal += bundle[amortization_type][rate_type][duration]['ppmt']
    tot_principal = abs(tot_principal)
    # print(tot_principal)

    for amortization_type in bundle.keys():
        for rate_type, rate in [('fixed', fixed_yearly_rate),
                                ('madad', madad_added_yearly_rate),
                                ('prime', prime_added_yearly_rate)]:  # TODO adjust prime to the content...
            if rate_type not in bundle[amortization_type].keys():
                continue
            for duration in bundle[amortization_type][rate_type].keys():
                d['amortization_type'].append(amortization_type)
                d['rate_type'].append(rate_type)
                d['nominal_rate'].append(rate)
                d['duration'].append(duration)
                d['principal'].append(abs(bundle[amortization_type][rate_type][duration]['ppmt']))
                d['principal_portion'].append(
                    abs(bundle[amortization_type][rate_type][duration]['ppmt']) / tot_principal)
                d['monthly_1st_payment'].append(abs(bundle[amortization_type][rate_type][duration]['pmt']))
                # d['monthly_max_payment'].append()
                d['interest_paid'].append(abs(bundle[amortization_type][rate_type][duration]['ipmt']))
                tot = d['principal'][-1] + d['interest_paid'][-1]
                d['net_paid'].append(tot)
                d['returned_ratio'].append(tot / abs(bundle[amortization_type][rate_type][duration]['ppmt']))
                d['effective_overall_rate'].append((d['returned_ratio'][-1] ** (1 / duration) - 1) * 12)

    df = pd.DataFrame.from_dict(d)
    return df

def optimize(asset_cost,
             capital,
             max_monthly_payment,
             # min_age, TODO...
             net_monthly_income,
             amortizations,
             is_married_couple=False,
             is_single_asset=True,
             prime=1):
    if prime == 2:
        prime_ = MAX_PRIME_PORTON
        use_max_prime = True
    elif prime == 1:
        prime_ = MAX_PRIME_PORTON / 2
        use_max_prime = True
    else:
        prime_ = MAX_PRIME_PORTON
        use_max_prime = False

    if (net_monthly_income * ALLOWABLE_MONTHLY_FRACTION < max_monthly_payment) or (max_monthly_payment is None):
        # TODO: raise a warning...
        # raise ValueError(f'maximum monthly payment is higher than allowed by the central bank ({net_monthly_income * ALLOWABLE_MONTHLY_FRACTION} < {max_monthly_payment})')
        max_monthly_payment = net_monthly_income * ALLOWABLE_MONTHLY_FRACTION

    loan_principal = asset_cost - capital
    funding_rate = loan_principal / asset_cost

    if is_single_asset:
        if funding_rate > MAX_FUNDING_RATE_FOR_FIRST_APPARTMENT:
            raise ValueError(
                f'funding rate is higher that allowed for single asset holding, by the central bank (!{funding_rate} > {MAX_FUNDING_RATE_FOR_FIRST_APPARTMENT})')
    else:
        if funding_rate > MAX_FUNDING_RATE_FOR_NON_FIRST_APPARTMENT:
            raise ValueError(
                f'funding rate is higher that allowed for non-single asset holding, by the central bank (!{funding_rate} > {MAX_FUNDING_RATE_FOR_NON_FIRST_APPARTMENT})')

    # determining interest rates
    # TODO: add interpolated ratings...

    if is_married_couple:
        fixed_yearly_rate = lambda r: interp1d([0., .45, .6, .7, .75], [.028, .0285, .03, .0315, .0315], kind='cubic')(r)
        madad_added_yearly_rate = lambda r: interp1d([0., .45, .6, .7, .75], [.018, .019, .0205, .0225, .0225],
                                                     kind='cubic')(r)
        prime_added_yearly_rate = -0.0064 * 1.1
    else:
        fixed_yearly_rate = lambda r: interp1d([0., .45, .6, .7, .75], [.03, .031, .033, .035, .035], kind='cubic')(r)
        madad_added_yearly_rate = lambda r: interp1d([0., .45, .6, .7, .75], [.0195, .0205, .0225, .0245, .0245],
                                                     kind='cubic')(r)
        prime_added_yearly_rate = -0.0064

    # print(fixed_yearly_rate, madad_added_yearly_rate, prime_added_yearly_rate)

    # TODO: incorporate age...
    bundle = optimize_mortgage(loan_principal,
                               max_monthly_payment,
                               amortizations,
                               use_max_prime=use_max_prime,
                               fixed_yearly_rate=fixed_yearly_rate(funding_rate),
                               madad_added_yearly_rate=madad_added_yearly_rate(funding_rate),
                               prime_added_yearly_rate=prime_added_yearly_rate,
                               increments=CALC_INCREMENTS,
                               max_prime_content=prime_)

    df = bundle_to_df(bundle,
                      fixed_yearly_rate=fixed_yearly_rate(funding_rate),
                      madad_added_yearly_rate=MADAD[0] + madad_added_yearly_rate(funding_rate),
                      prime_added_yearly_rate=PRIME[0] + prime_added_yearly_rate)

    # else:
    #     raise ValueError(
    #         f'at least one method of amortization should be considered (Shpitzer or, and Equal)')
    #

    return df







if __name__ == '__main__':

    asset_cost = 100000
    capital = 25000
    max_monthly_payment = 3600
    net_monthly_income = 1200
    is_married_couple = False
    is_single_asset = True
    amortizations = ['shpitzer']  # , 'equal']
    prime = 1 # 33%

    from time import time
    tic = time()
    df = optimize(asset_cost,
                 capital,
                 max_monthly_payment,
                 net_monthly_income,
                 amortizations,
                 is_married_couple=is_married_couple,
                 is_single_asset=is_single_asset,
                 prime=prime)

    for c in df.columns:
        print(df[c])


