import numpy as np
import pandas as pd
from financials import update_yearly_to_monthly_rates_with_risk

make_float = lambda x: '{:,.2f}'.format(x)
make_int = lambda x: '{:,}'.format(int(x))


def input_args_to_df(asset_cost,
                     capital,
                     max_monthly_payment,
                     net_monthly_income,
                     is_married_couple,
                     is_single_asset,
                     amortizations,
                     prime):

    if is_married_couple:
        is_married_couple = 'זוג'
    else:
        is_married_couple = 'יחיד'

    if is_single_asset:
        is_single_asset = 'כן'
    else:
        is_single_asset = 'לא'

    if len(amortizations) > 1:
        amortizations = 'שפיצר + קרן-שווה'
    elif amortizations[0] == 'spitzer':
        amortizations = 'שפיצר'
    else:
        amortizations = 'קרן-שווה'

    if prime == 2:
        prime = 'מקסימלית - % 66'
    elif prime == 1:
        prime = 'מקסימלי ישן - % 33'
    else:
        prime = 'חלוקה מיטבית אוטומטית'

    input_params = [
        [make_int(asset_cost) + ' ₪', 'מחיר הנכס'],
        [make_int(capital) + ' ₪', 'הון עצמי'],
        [make_int(net_monthly_income) + ' ₪', 'הכנסה חודשית נטו'],
        [make_int(max_monthly_payment) + ' ₪', 'תשלום חודשי מקסימלי'],
        [is_married_couple, 'מצב משפחתי'],
        [is_single_asset, 'דירה יחידה'],
        [amortizations, 'לוחות סילוקין לחישוב'],
        [prime, 'תכולת פריים רצויה'],
        [make_int(asset_cost - capital) + ' ₪', 'גודל הקרן'],
        [make_float(100 * (asset_cost - capital) / asset_cost) + ' %', 'אחוז מימון'],
    ]
    df = pd.DataFrame(input_params, columns=['בחירה (קלט)', 'פרמטר'])
    return df


def stylish_html(df, title=''):
    result = ''
    result += '<h4> %s </h4>\n' % title
    result += df.to_html(classes='wide', escape=False, index=False)
    result += '''
    </body>
    </html>
    '''
    return result


def df_to_summary_df(total_monthly_payments):
    df = pd.DataFrame(data=[[make_float(total_monthly_payments['total']['pmt'].sum() /
                                        total_monthly_payments['total']['ppmt'].sum()),
                            make_int(total_monthly_payments['total']['pmt'].sum()),
                            make_int(total_monthly_payments['total']['ipmt'].sum()),
                            make_int(total_monthly_payments['total']['pmt'][0]),
                            make_float(100 * (total_monthly_payments['total']['pmt'].sum() /
                                        total_monthly_payments['total']['ppmt'].sum()) /
                                       len(total_monthly_payments['total']['pmt'])),
                            make_int(len(total_monthly_payments['total']['pmt'])),
                            make_int(total_monthly_payments['total']['ppmt'].sum())]],
                      columns=['מחיר משוקלל לשקל',
                               'סה"כ לתשלום ₪',
                               'סה"כ עלות ₪',
                               'תשלום ראשון ₪',
                               'ריבית משוקללת %',
                               'חודשי תשלום',
                               'סך הקרן ₪'])
    return df


def beutify_HTML(total_monthly_payments,
                 asset_cost,
                 capital,
                 max_monthly_payment,
                 net_monthly_income,
                 is_married_couple,
                 is_single_asset,
                 amortizations,
                 prime):
    principal = asset_cost - capital
    summay_df = df_to_summary_df(total_monthly_payments)
    funding_rate = principal / asset_cost
    monthly_rates = update_yearly_to_monthly_rates_with_risk(funding_rate, is_married_couple)

    d = {
        'amortization_type': [],
        'rate_type': [],
        'nominal_rate': [],
        'duration': [],
        'monthly_1st_payment': [],
        # 'monthly_max_payment': '',
        'principal': [],
        'principal_portion': [],
        'interest_paid': [],
        'net_paid': [],
        'returned_ratio': [],
        # 'effective_overall_rate': []
    }

    for k in total_monthly_payments.keys():
        if 'total' in k:
            continue

        for a in ['prime', 'madad', 'fixed']:
            if a in k:
                d['rate_type'].append(a)
                for rate in monthly_rates.keys():
                    if a in rate:
                        d['nominal_rate'].append(make_float(12 * 100 * np.average(monthly_rates[rate])))
                        break
                break

        if len(amortizations) == 1:
            d['amortization_type'].append(amortizations[0])
        else:
            for a in amortizations:
                if a in k:
                    d['amortization_type'].append(a)
                    break

        d['duration'].append(k.split('_')[-1])
        d['monthly_1st_payment'].append(make_int(total_monthly_payments[k]['pmt'][0]))
        d['principal'].append(make_int(total_monthly_payments[k]['ppmt'].sum()))
        d['principal_portion'].append(make_float(100 * total_monthly_payments[k]['ppmt'].sum() / principal))
        d['interest_paid'].append(make_int(total_monthly_payments[k]['ipmt'].sum()))
        d['net_paid'].append(make_int(total_monthly_payments[k]['pmt'].sum()))
        d['returned_ratio'].append(make_float(total_monthly_payments[k]['pmt'].sum() / total_monthly_payments[k]['ppmt'].sum()))

    df = pd.DataFrame.from_dict(d)

    summay_df[summay_df.columns[3]] = make_int(max_monthly_payment)
    summay_df[summay_df.columns[-1]] = make_int(principal)
    summay_df[summay_df.columns[-3]] = make_float(sum((df['principal_portion'].values.astype('float32') / 100) *
                                       df['nominal_rate'].values.astype('float32')))

    # Hebrewfy:
    Hebs = {
        'amortization_type': 'לוח סילוקין',
        'rate_type': 'סוג ריבית',
        'nominal_rate': 'ריבית %',
        'duration': 'חודשי תשלום',
        'monthly_1st_payment': 'תשלום ראשון ₪',
        # 'monthly_max_payment': '',
        'principal': 'קרן ₪',
        'principal_portion': 'מרכיב קרן %',
        'interest_paid': 'סה"כ עלות ₪',
        'net_paid': 'סה"כ תשלום ₪',
        'returned_ratio': 'מחיר לשקל ₪',
        'effective_overall_rate': 'ריבית אפקטיבית %',

        'spitzer': 'שפיצר',
        'equal': 'קרן-שווה',

        'fixed': 'קל"צ',
        'madad': 'ק"צ',
        'prime': 'פריים'
    }

    df = df.replace(Hebs)
    df = df.rename(columns=Hebs)

    # design:
    style = '''
        <html>
        <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="stylesheet" href="https://pyscript.net/latest/pyscript.css" />
        <style>

    		body {
    			background-color: rgb(255,255,255);
    		}
            h2 {
                text-align: center;
                font-family: Helvetica, Arial, sans-serif;
            }
            h3 {
                text-align: center;
                font-family: Helvetica, Arial, sans-serif;
            }
            h4 {
                text-align: center;
                font-family: Helvetica, Arial, sans-serif;
            }
            h6 {
                text-align: center;
                font-family: Helvetica, Arial, sans-serif;
            }
            table { 
    			align: center;
                margin-left: auto;
                margin-right: auto;
    			background-color: rgb(255,255,255);
            }
            table, th, td {
                border: 1px solid black;
                border-collapse: collapse;
            }
            th, td {
                padding: 5px;
                text-align: center;
                font-family: Helvetica, Arial, sans-serif;
                font-size: 80%;
            }
            table tbody tr:hover {
                background-color: #dddddd;
            }

        </style>
        </head>
        <body>
        '''

    # header:
    header = style
    header += f'<h2> {"תמהיל נבחר"} </h2>'
    header += '''
            </body>
            </html>
        '''

    df = stylish_html(df, title='תמהיל אופטמלי') + stylish_html(summay_df, title='סיכום')
    # if input_args is not None:
    input_df = input_args_to_df(asset_cost,
                                 capital,
                                 max_monthly_payment,
                                 net_monthly_income,
                                 is_married_couple,
                                 is_single_asset,
                                 amortizations,
                                 prime)
    input_df = stylish_html(input_df, title='נתוני משתמש')
    df = input_df + df

    # footer:
    footer = f'<h6> {".אין לראות את האתר והכלים בו כהמלצה פיננסית מכל סוג, ויוצריו אינם אחראיים לצעדי המשתמשים בהקשר זה"} </h6>'
    footer += f'<h6> {":כל הזכויות של&nbsp;כלי חישוב המשכנתא&nbsp;שמורות ליוצר"} </h6>'
    footer += f'<h6><a href="http://linkedin.com/in/natanel-davidovits-28695312" target="_blank" rel="noopener"> {"נתנאל דוידוביץ"} </a></h6>\n'
    footer += '''
            </body>
            </html>
        '''
    page = header + df + footer

    return page

if __name__ == '__main__':
    ...