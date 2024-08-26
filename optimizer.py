from scipy.optimize import differential_evolution, NonlinearConstraint

try:
    import requests
except:
    pass
    # import pyodide_js
    # await pyodide_js.loadPackage('requests')

from financials import *

import warnings
warnings.filterwarnings('ignore')


def monthly_payment_bank(monthly_rate: np.ndarray, equal_amortization=False):
    if equal_amortization:
        amortization_function = get_equal_amortization
    else:
        amortization_function = get_spitzer_amortization
    return dict(map(lambda n: (n, amortization_function(monthly_rate, n, 1)), DURATIONS))


def tri_amortization_composition_duration_variable(monthly_rates_dictionary: dict, equal_amortization=False) -> dict:
    payments_bank = {}
    for amortization in ['prime', 'madad', 'fixed']:
        payments_bank[f'{amortization}_monthly_payments'] = (
            monthly_payment_bank(monthly_rates_dictionary[f'{amortization}_rate'], equal_amortization))
    return payments_bank


def get_optimized_composition(monthly_rates_dictionary: dict,
                              principal_portions: dict,
                              max_first_payment_fraction: float,
                              equal_amortization=False) -> (dict, float):
    # principal_portions = {'fixed': 0.6, 'madad': 0.6, 'prime': 0.6}
    if not all(list(map(lambda x: (type(x) is float) or (type(x) is np.float_), list(principal_portions.values())))):
        raise ValueError('>>> type(values) != float')
    if sum(principal_portions.values()) != 1:
        raise ValueError('>>> sum(values) != 1')
    payments_bank = tri_amortization_composition_duration_variable(monthly_rates_dictionary,
                                                                   equal_amortization=equal_amortization)
    # presumming, for later efficiency:
    _payments_bank = {}
    for monthly_payments in payments_bank.keys():
        _payments_bank[monthly_payments] = {}
        for duration in payments_bank[monthly_payments].keys():
            _payments_bank[monthly_payments][duration] = (payments_bank[monthly_payments][duration]['pmt'].sum() *
                                                          principal_portions[monthly_payments.split('_')[0]])

    net_payments = np.inf
    optimal_result = {}
    for d1 in DURATIONS: # fixed_monthly_payments
        for d2 in DURATIONS: # madad_monthly_payments
            for d3 in DURATIONS: # prime_monthly_payments
                tot = (_payments_bank['fixed_monthly_payments'][d1] +
                       _payments_bank['madad_monthly_payments'][d2] +
                       _payments_bank['prime_monthly_payments'][d3])
                if tot < net_payments:
                    fixed_pmt = payments_bank['fixed_monthly_payments'][d1]['pmt'] * principal_portions['fixed']
                    madad_pmt = payments_bank['madad_monthly_payments'][d2]['pmt'] * principal_portions['madad']
                    prime_pmt = payments_bank['prime_monthly_payments'][d3]['pmt'] * principal_portions['prime']
                    if max_first_payment_fraction < fixed_pmt[0] + madad_pmt[0] + prime_pmt[0]:
                        continue
                    fixed, madad, prime = {}, {}, {}
                    for p in ['pmt', 'ipmt', 'ppmt']:
                        fixed[p] = payments_bank['fixed_monthly_payments'][d1][p] * principal_portions['fixed']
                        madad[p] = payments_bank['madad_monthly_payments'][d2][p] * principal_portions['madad']
                        prime[p] = payments_bank['prime_monthly_payments'][d3][p] * principal_portions['prime']

                    optimal_result = {'fixed_monthly_payments': {d1: fixed},
                                      'madad_monthly_payments': {d2: madad},
                                      'prime_monthly_payments': {d3: prime}}
                    net_payments = tot
    return optimal_result, net_payments


def get_optimized_principal_portions(monthly_rates_dictionary: dict,
                                     max_first_payment_fraction: float,
                                     equal_amortization=False,
                                     set_prime_portion=None) -> (dict, float):
    if set_prime_portion is None:
        def target_function(portions_array): # (fixed_portion, madad_portion, prime_portion)
            try:
                _, net_payments = get_optimized_composition(monthly_rates_dictionary,
                                                            {'fixed': portions_array[0],
                                                             'madad': portions_array[1],
                                                             'prime': portions_array[2]},
                                                            max_first_payment_fraction,
                                                            equal_amortization=equal_amortization)
            except ValueError:
                return np.pi # >3 x 1 (principal)
            return net_payments

        bounds = [(MIN_FIXED_PORTION, 1),
                  (MIN_UNFIXED_PORTON, MAX_UNFIXED_PORTON),
                  (MIN_UNFIXED_PORTON, MAX_UNFIXED_PORTON)]
        constraints = (NonlinearConstraint(lambda x: x.sum(), 1, 1))
        result = differential_evolution(target_function, bounds, constraints=constraints, disp=False,seed=SEED)
        optimal_principal_portions = {'fixed': result.x[0], 'madad': result.x[1], 'prime': result.x[2]}
    else:
        def target_function(portions_array):  # (fixed_portion, madad_portion)
            try:
                _, net_payments = get_optimized_composition(monthly_rates_dictionary,
                                                            {'fixed': portions_array[0],
                                                             'madad': portions_array[1],
                                                             'prime': set_prime_portion},
                                                            max_first_payment_fraction,
                                                            equal_amortization=equal_amortization)
            except ValueError:
                return np.pi # >3 x 1 (principal)
            return net_payments

        bounds = [(MIN_FIXED_PORTION, 1),
                  (MIN_UNFIXED_PORTON, MAX_UNFIXED_PORTON - set_prime_portion)]
        constraints = (NonlinearConstraint(lambda x: x.sum(), 1 - set_prime_portion, 1 - set_prime_portion))
        result = differential_evolution(target_function, bounds, constraints=constraints, disp=False, seed=SEED)
        optimal_principal_portions = {'fixed': result.x[0], 'madad': result.x[1], 'prime': set_prime_portion}

    return get_optimized_composition(monthly_rates_dictionary,
                                     optimal_principal_portions,
                                     max_first_payment_fraction,
                                     equal_amortization=equal_amortization)


def get_optimized_principal_portions_with_amortization_defined(monthly_rates_dictionary: dict,
                                                               max_first_payment_fraction: float,
                                                               equal_amortization=None,
                                                               set_prime_portion=None) -> (dict, float):
    if equal_amortization is not None:
        return get_optimized_principal_portions(monthly_rates_dictionary,
                                                max_first_payment_fraction,
                                                equal_amortization=equal_amortization,
                                                set_prime_portion=set_prime_portion)

    equal_optimal_result, equal_net_payments = get_optimized_principal_portions(monthly_rates_dictionary,
                                                                                max_first_payment_fraction,
                                                                               equal_amortization=True,
                                                                               set_prime_portion=set_prime_portion)
    spitzer_optimal_result, spitzer_net_payments = get_optimized_principal_portions(monthly_rates_dictionary,
                                                                                    max_first_payment_fraction,
                                                                                   equal_amortization=False,
                                                                                   set_prime_portion=set_prime_portion)

    def target_function(equal_portion):
        return equal_portion * equal_net_payments + (1 - equal_portion) * spitzer_net_payments

    bounds = [(0, 1)]
    result = differential_evolution(target_function, bounds, disp=False, seed=SEED)
    equal_portion = result.x
    if equal_portion >= 0.97:
        _equal_optimal_result = {}
        for k in equal_optimal_result.keys():
            _equal_optimal_result['equal_' + k] = equal_optimal_result[k]
        return _equal_optimal_result, equal_net_payments
    elif equal_portion <= 0.03:
        _spitzer_optimal_result = {}
        for k in equal_optimal_result.keys():
            _spitzer_optimal_result['spitzer_' + k] = spitzer_optimal_result[k]
        return _spitzer_optimal_result, spitzer_net_payments

    for k in equal_optimal_result.keys():
        for d in equal_optimal_result[k].keys():
            equal_optimal_result[k][d]['pmt'] = equal_optimal_result[k][d]['pmt'] * equal_portion
            equal_optimal_result[k][d]['ipmt'] = equal_optimal_result[k][d]['ipmt'] * equal_portion
            equal_optimal_result[k][d]['ppmt'] = equal_optimal_result[k][d]['ppmt'] * equal_portion

    spitzer_portion = 1 - equal_portion
    for k in spitzer_optimal_result.keys():
        for d in spitzer_optimal_result[k].keys():
            spitzer_optimal_result[k][d]['pmt'] = spitzer_optimal_result[k][d]['pmt'] * spitzer_portion
            spitzer_optimal_result[k][d]['ipmt'] = spitzer_optimal_result[k][d]['ipmt'] * spitzer_portion
            spitzer_optimal_result[k][d]['ppmt'] = spitzer_optimal_result[k][d]['ppmt'] * spitzer_portion

    net_payments = equal_net_payments * equal_portion + spitzer_net_payments * spitzer_portion

    optimal_result = {'equal_optimal_result': equal_optimal_result, 'spitzer_optimal_result': spitzer_optimal_result}
    return optimal_result, net_payments

def optimize(max_first_payment_fraction: float,
             funding_rate: float,
             is_married_couple=False,
             equal_amortization=None,
             set_prime_portion=None):
    if max_first_payment_fraction > 1 / MIN_DURATION:
        raise ValueError(f'Max first payment fraction must be <= 1 / {MIN_DURATION} '
                         f'>>> max_first_payment_fraction == {max_first_payment_fraction}.')
    if max_first_payment_fraction < 1 / MAX_DURATION:
        raise ValueError(f'Max first payment fraction must be >= 1 / {MAX_DURATION} '
                         f'>>> max_first_payment_fraction == {max_first_payment_fraction}.')
    if funding_rate > MAX_FUNDING_RATE_FOR_FIRST_APPARTMENT:
        raise ValueError(f'Max funding rate must be <= {MAX_FUNDING_RATE_FOR_FIRST_APPARTMENT} '
                         f'>>> funding_rate == {funding_rate}.')
    if funding_rate <= 0:
        raise ValueError(f'Max funding rate must be > {0} '
                         f'>>> funding_rate == {funding_rate}.')
    monthly_rates = update_yearly_to_monthly_rates_with_risk(funding_rate, is_married_couple)
    return get_optimized_principal_portions_with_amortization_defined(monthly_rates,
                                                                      max_first_payment_fraction,
                                                                      equal_amortization=equal_amortization,
                                                                      set_prime_portion=set_prime_portion)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    from pprint import PrettyPrinter
    pp = PrettyPrinter().pprint

    asset_cost = 2150000
    capital = 950000
    max_monthly_payment = 20000
    net_monthly_income = 6000
    is_married_couple = False
    equal_amortization = True

    # funding_rate = 0.75
    # principal_portions = {'madad': 0.37, 'prime': 0.33, 'fixed': 0.3}
    # d = update_yearly_to_monthly_rates_with_risk(funding_rate, is_married_couple)

    # payments_bank = tri_amortization_composition_duration_variable(d, principal_portions, equal_amortization=False)
    # pp(payments_bank)

    # optimal_result, net_payments = get_optimized_composition(d, principal_portions, equal_amortization=True)
    # pp(optimal_result)

    # optimal_result, net_payments = get_optimized_principal_portions(d, equal_amortization=True, set_prime_portion=None)
    # pp(optimal_result)

    # optimal_result, net_payments = get_optimized_principal_portions_with_amortization_defined(d,
    #                                                                                           max_first_payment_fraction,
    #                                                                                           equal_amortization=None,
    #                                                                                           set_prime_portion=None)
    # pp(optimal_result)

    set_prime_portion = 0.333
    principal = asset_cost - capital
    max_first_payment_fraction = max_monthly_payment / principal
    funding_rate = principal / asset_cost
    optimal_result, net_payments = optimize(max_first_payment_fraction,
                                            funding_rate,
                                            is_married_couple=is_married_couple,
                                            equal_amortization=equal_amortization,
                                            set_prime_portion=set_prime_portion)
    pp(optimal_result)

    # multiplying with finance:
    optimal_result_finance = {}
    for k1, v in optimal_result.items():
        k2 = list(v.keys())[0]
        optimal_result_finance[k1] = np.ceil(v[k2]['pmt'] * principal).astype('int32')
        print(optimal_result_finance[k1].shape)

    print('-'*44)
    pp(optimal_result_finance)

    # array_length = []
    # for k1, v in optimal_result.items():
    #     k2 = list(v.keys())[0]
    #     array_length.append(len(v[k2]))
    # tot = np.zeros(max(array_length))
    # for k1, v in optimal_result.items():
    #     k2 = list(v.keys())[0]
    #     tot[:len(v[k2])] = tot[:len(v[k2])] + v[k2]
    #     v[k2] = np.append(v[k2], [0])
    #     plt.plot(v[k2], label=f'{k1.replace("_", " ")}: {k2} months')
    # tot = np.append(tot, [0])
    # plt.plot(tot, label=f'sum of all amortizations')
    # plt.legend()
    # plt.xlabel('month')
    # plt.ylabel('payment')
    # plt.grid(True)
    # plt.show()


