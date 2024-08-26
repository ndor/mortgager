from optimizer import optimize
from htmling import *

try:
    from pyscript import document, window
except (ModuleNotFoundError, ImportError):
    pass

import warnings
warnings.filterwarnings('ignore')



from pprint import PrettyPrinter
pp = PrettyPrinter().pprint


def func(event):
    asset_cost = int(document.querySelector("#asset_cost").value)
    capital = int(document.querySelector("#capital").value)
    net_monthly_income = int(document.querySelector("#net_monthly_income").value)
    max_monthly_payment = int(document.querySelector("#max_monthly_payment").value)

    amortizations = document.querySelectorAll("[name='amortizations']:checked")
    amortizations = list(map(lambda i: amortizations[i].value, range(len(amortizations))))
    equal_amortization = False
    if len(amortizations) > 1:
        equal_amortization = None
    elif 'equal' in amortizations:
        equal_amortization = True

    is_married_couple = not (document.querySelector("[name='is_married_couple']:checked") == None)
    is_single_asset = not (document.querySelector("[name='is_single_asset']:checked") == None)

    prime = int(document.querySelector("[name='prime']:checked").value)
    if prime == 0:
        _prime = None
    elif prime == 1:
        _prime = 1 / 3
    else:
        _prime = 2 / 3

    # mortgage calculation:
    principal = asset_cost - capital
    max_first_payment_fraction = max_monthly_payment / principal
    funding_rate = principal / asset_cost
    optimal_result, net_payments = optimize(max_first_payment_fraction,
                                            funding_rate,
                                            is_married_couple=is_married_couple,
                                            equal_amortization=equal_amortization,
                                            set_prime_portion=_prime)

    # weighted interest:
        # flattening the dict:
    _optimal_result = {}
    if any(map(lambda x: any([(y in x) for y in ['prime', 'madad', 'fixed']]), optimal_result.keys())): # single type of amortization
        for k1 in optimal_result.keys():
            duration = list(optimal_result[k1].keys())[0]
            _optimal_result[f'{k1}_{duration}'] = optimal_result[k1][duration]
    else:
        for k1 in ['equal_optimal_result', 'spitzer_optimal_result']:
            for k2 in optimal_result[k1].keys():
                duration = list(optimal_result[k1][k2].keys())[0]
                _optimal_result[f'{k1.split("_")[0]}_{k2}_{duration}'] = optimal_result[k1][k2][duration]
    optimal_result = _optimal_result
    del _optimal_result

    # adding the sum (tot) of amortizations:
    array_length = []
    for k1, v in optimal_result.items():
        array_length.append(int(k1.split('_')[-1]))
    tot_pmt = np.zeros(max(array_length))
    tot_ipmt = np.zeros(max(array_length))
    tot_ppmt = np.zeros(max(array_length))
    for k1, v in optimal_result.items():
        duration = int(k1.split('_')[-1])
        tot_pmt[:duration] = tot_pmt[:duration] + v['pmt']
        tot_ipmt[:duration] = tot_ipmt[:duration] + v['ipmt']
        tot_ppmt[:duration] = tot_ppmt[:duration] + v['ppmt']
    optimal_result['total'] = {'pmt': tot_pmt, 'ipmt': tot_ipmt, 'ppmt': tot_ppmt}

    # updating total_monthly_payments to real cost:
    total_monthly_payments = {}
    for k1 in optimal_result.keys():
        total_monthly_payments[k1] = {}
        for p in ['pmt', 'ipmt', 'ppmt']:
            total_monthly_payments[k1][p] = np.ceil(optimal_result[k1][p] * principal).astype('int32')

    html_out = beutify_HTML(total_monthly_payments,
                            asset_cost,
                            capital,
                            max_monthly_payment,
                            net_monthly_income,
                            is_married_couple,
                            is_single_asset,
                            amortizations,
                            prime)

    new = window.open()
    new.document.body.innerHTML = html_out

# # for testing
# def func(event):
#     asset_cost = 2150000 #int(document.querySelector("#asset_cost").value)
#     capital = 950000 #int(document.querySelector("#capital").value)
#     net_monthly_income = 20154 #int(document.querySelector("#net_monthly_income").value)
#     max_monthly_payment = 6718 #int(document.querySelector("#max_monthly_payment").value)
#
#     # amortizations = document.querySelectorAll("[name='amortizations']:checked")
#     amortizations = ['equal', 'spitzer']#list(map(lambda i: amortizations[i].value, range(len(amortizations))))
#     equal_amortization = False
#     if len(amortizations) > 1:
#         equal_amortization = None
#     elif 'equal' in amortizations:
#         equal_amortization = True
#
#     is_married_couple = False #not (document.querySelector("[name='is_married_couple']:checked") == None)
#     is_single_asset = True #not (document.querySelector("[name='is_single_asset']:checked") == None)
#
#     prime = 0 #int(document.querySelector("[name='prime']:checked").value)
#     if prime == 0:
#         _prime = None
#     elif prime == 1:
#         _prime = 1 / 3
#     else:
#         _prime = 2 / 3
#
#     # mortgage calculation:
#     principal = asset_cost - capital
#     max_first_payment_fraction = max_monthly_payment / principal
#     funding_rate = principal / asset_cost
#     optimal_result, net_payments = optimize(max_first_payment_fraction,
#                                             funding_rate,
#                                             is_married_couple=is_married_couple,
#                                             equal_amortization=equal_amortization,
#                                             set_prime_portion=_prime)
#
#     # weighted interest:
#         # flattening the dict:
#     _optimal_result = {}
#     if any(map(lambda x: any([(y in x) for y in ['prime', 'madad', 'fixed']]), optimal_result.keys())): # single type of amortization
#         for k1 in optimal_result.keys():
#             duration = list(optimal_result[k1].keys())[0]
#             _optimal_result[f'{k1}_{duration}'] = optimal_result[k1][duration]
#     else:
#         for k1 in ['equal_optimal_result', 'spitzer_optimal_result']:
#             for k2 in optimal_result[k1].keys():
#                 duration = list(optimal_result[k1][k2].keys())[0]
#                 _optimal_result[f'{k1.split("_")[0]}_{k2}_{duration}'] = optimal_result[k1][k2][duration]
#     optimal_result = _optimal_result
#     del _optimal_result
#
#     # adding the sum (tot) of amortizations:
#     array_length = []
#     for k1, v in optimal_result.items():
#         array_length.append(int(k1.split('_')[-1]))
#     tot_pmt = np.zeros(max(array_length))
#     tot_ipmt = np.zeros(max(array_length))
#     tot_ppmt = np.zeros(max(array_length))
#     for k1, v in optimal_result.items():
#         duration = int(k1.split('_')[-1])
#         tot_pmt[:duration] = tot_pmt[:duration] + v['pmt']
#         tot_ipmt[:duration] = tot_ipmt[:duration] + v['ipmt']
#         tot_ppmt[:duration] = tot_ppmt[:duration] + v['ppmt']
#     optimal_result['total'] = {'pmt': tot_pmt, 'ipmt': tot_ipmt, 'ppmt': tot_ppmt}
#
#     # updating total_monthly_payments to real cost:
#     total_monthly_payments = {}
#     for k1 in optimal_result.keys():
#         total_monthly_payments[k1] = {}
#         for p in ['pmt', 'ipmt', 'ppmt']:
#             total_monthly_payments[k1][p] = np.ceil(optimal_result[k1][p] * principal).astype('int32')
#
#     html_out = beutify_HTML(total_monthly_payments,
#                             asset_cost,
#                             capital,
#                             max_monthly_payment,
#                             net_monthly_income,
#                             is_married_couple,
#                             is_single_asset,
#                             amortizations,
#                             prime)
#     print(html_out)
#     # new = window.open()
#     # new.document.body.innerHTML = html_out



if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # from pprint import PrettyPrinter
    # pp = PrettyPrinter().pprint
    #
    # asset_cost = 2150000
    # capital = 950000
    # max_monthly_payment = 20000
    # net_monthly_income = 6000
    # is_married_couple = False
    # equal_amortization = False
    # set_prime_portion = 0.333
    # principal = asset_cost - capital
    # max_first_payment_fraction = max_monthly_payment / principal
    # funding_rate = principal / asset_cost
    #
    # optimal_result, net_payments = optimize(max_first_payment_fraction,
    #                                         funding_rate,
    #                                         is_married_couple=is_married_couple,
    #                                         equal_amortization=equal_amortization,
    #                                         set_prime_portion=set_prime_portion)
    # pp(optimal_result)
    #
    # # multiplying with finance:
    # optimal_result_finance = {}
    # for k1, v in optimal_result.items():
    #     k2 = list(v.keys())[0]
    #     optimal_result_finance[k1] = np.ceil(v[k2] * principal).astype('int32')
    #     print(optimal_result_finance[k1].shape)
    # print('-'*44)
    # pp(optimal_result_finance)
    #
    # # array_length = []
    # # for k1, v in optimal_result.items():
    # #     k2 = list(v.keys())[0]
    # #     array_length.append(len(v[k2]))
    # # tot = np.zeros(max(array_length))
    # # for k1, v in optimal_result.items():
    # #     k2 = list(v.keys())[0]
    # #     tot[:len(v[k2])] = tot[:len(v[k2])] + v[k2]
    # #     v[k2] = np.append(v[k2], [0])
    # #     plt.plot(v[k2], label=f'{k1.replace("_", " ")}: {k2} months')
    # # tot = np.append(tot, [0])
    # # plt.plot(tot, label=f'sum of all amortizations')
    # # plt.legend()
    # # plt.xlabel('month')
    # # plt.ylabel('payment')
    # # plt.grid(True)
    # # plt.show()
    #
    #
    #
    #
    # html_out = beutify_HTML(df, asset_cost,
    #                              capital,
    #                              max_monthly_payment,
    #                              net_monthly_income,
    #                              is_married_couple,
    #                              is_single_asset,
    #                              amortizations,
    #                              prime)
    #
    # print(df)
    # print(html_out)
    # print(time() - tic)


    func(None)


