import numpy as np
import pandas as pd


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
    elif amortizations[0] == 'shpitzer':
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
    '''
    Write an entire dataframe to an HTML file with nice formatting.
    '''
    result = ''
    result += '<h4> %s </h4>\n' % title
    result += df.to_html(classes='wide', escape=False, index=False)
    result += '''
    </body>
    </html>
    '''
    return result


def df_to_summay_df(df):
    df = pd.DataFrame(data=[[make_float(df['net_paid'].sum() / df['principal'].sum()),
                            make_int(df['net_paid'].sum()),
                            make_int(df['interest_paid'].sum()),
                            make_int(df['monthly_1st_payment'].sum()),
                            make_float(100 * (df['nominal_rate'] * df['principal_portion']).sum()),
                            make_int(df['duration'].max()),
                            make_int(df['principal'].sum())]],
                      columns=['עלות משוקללת ל 1₪',
                               'סה"כ לתשלום ₪',
                               'סה"כ עלות ₪',
                               'תשלום ראשון ₪',
                               'ריבית משוקללת %',
                               'חודשי תשלום',
                               'סך הקרן ₪'])
    return df


def beutify_HTML(df,
                 asset_cost,
                 capital,
                 max_monthly_payment,
                 net_monthly_income,
                 is_married_couple,
                 is_single_asset,
                 amortizations,
                 prime,
                 background_color='rgb(255, 255, 255)'):
    summay_df = df_to_summay_df(df)
    df = df.drop(['effective_overall_rate'], axis=1)
    percent_fields = ['nominal_rate', 'principal_portion']  # , 'effective_overall_rate']#, 'returned_ratio'
    for c in df.columns:
        if c in percent_fields:
            df[c] = np.around(df[c].astype(float) * 100, decimals=2)
            df[c] = df[c].apply(make_float)
        elif type(df[c].iloc[0]) != str:
            if c == 'returned_ratio':
                df[c] = np.around(df[c], decimals=2)
                df[c] = df[c].apply(make_float)
            else:
                df[c] = df[c].astype('int').apply(make_int)

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

        'shpitzer': 'שפיצר',
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
    footer += f'<h6> {":כל הזכויות של&nbsp;כלי חישוב המשכנתא&nbsp; שמורות ליוצר"} </h6>'
    footer += f'<h6><a href="http://linkedin.com/in/natanel-davidovits-28695312" target="_blank" rel="noopener"> {"נתנאל דוידוביץ"} </a></h6>\n'
    footer += '''
            </body>
            </html>
        '''
    df = header + df + footer
    return df

if __name__ == '__main__':
    ...