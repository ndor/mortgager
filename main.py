from optimizer import optimize
from htmling import *

try:
    from pyscript import document, window
except ModuleNotFoundError:
    pass

import warnings
warnings.filterwarnings('ignore')

def func(event):
    asset_cost = int(document.querySelector("#asset_cost").value)
    capital = int(document.querySelector("#capital").value)
    net_monthly_income = int(document.querySelector("#net_monthly_income").value)
    max_monthly_payment = int(document.querySelector("#max_monthly_payment").value)

    amortizations = document.querySelectorAll("[name='amortizations']:checked")
    amortizations = list(map(lambda i: amortizations[i].value, range(len(amortizations))))

    is_married_couple = not (document.querySelector("[name='is_married_couple']:checked") == None)
    is_single_asset = not (document.querySelector("[name='is_single_asset']:checked") == None)
    prime = int(document.querySelector("[name='prime']:checked").value)

    df = optimize(asset_cost,
                 capital,
                 max_monthly_payment,
                 net_monthly_income,
                 amortizations,
                 is_married_couple=is_married_couple,
                 is_single_asset=is_single_asset,
                 prime=prime)

    html_out = beutify_HTML(df,
                            asset_cost,
                            capital,
                            max_monthly_payment,
                            net_monthly_income,
                            is_married_couple,
                            is_single_asset,
                            amortizations,
                            prime,
                            background_color='rgb(255, 255, 255)')

    new = window.open()
    new.document.body.innerHTML = html_out





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

    html_out = beutify_HTML(df, asset_cost,
                                 capital,
                                 max_monthly_payment,
                                 net_monthly_income,
                                 is_married_couple,
                                 is_single_asset,
                                 amortizations,
                                 prime)

    print(df)
    print(html_out)
    print(time() - tic)


