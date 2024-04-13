import copy
import numpy as np
# import numpy_financial as npf
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, brute, shgo, minimize, LinearConstraint, NonlinearConstraint

try:
    import requests
except:
    pass
    # import pyodide_js
    # await pyodide_js.loadPackage('requests')

try:
    from pyscript import document, window
except ModuleNotFoundError:
    pass
# from pyscript import document, window

import warnings
warnings.filterwarnings('ignore')

'''
http://www.yorku.ca/amarshal/mortgage.htm
https://www.moti.org.il/intrests
'''

SEED = 108
CALC_INCREMENTS = 10
dt_m = 12
dt_y = 5
MAX_PRIME_PORTON = 0.666
MINIMAL_FIXED_PORTION = 0.333
MAX_FUNDING_RATE_FOR_FIRST_APPARTMENT = 0.75
MAX_FUNDING_RATE_FOR_NON_FIRST_APPARTMENT = 0.5
MAX_DURATION = 30 * dt_m
MIN_DURATION = 5 * dt_m
ALLOWABLE_MONTHLY_FRACTION = 0.334
BANK_BASE = {'shpitzer': {'fixed': {}, 'madad': {}, 'prime': {}}, 'equal': {'fixed': {}, 'madad': {}, 'prime': {}}}
DURATIONS = list(range(MIN_DURATION, MAX_DURATION + dt_m * dt_y, dt_m * dt_y))  # duration in step, by years strides

make_float = lambda x: '{:,.2f}'.format(x)
make_int = lambda x: '{:,}'.format(int(x))

_when_to_num = {'end': 0, 'begin': 1,
                'e': 0, 'b': 1,
                0: 0, 1: 1,
                'beginning': 1,
                'start': 1,
                'finish': 0}

def _convert_when(when):
    # Test to see if when has already been converted to ndarray
    # This will happen if one function calls another, for example ppmt
    if isinstance(when, np.ndarray):
        return when
    try:
        return _when_to_num[when]
    except (KeyError, TypeError):
        return [_when_to_num[x] for x in when]

def npf_pmt(rate, nper, pv, fv=0, when='end'):
    """
    Compute the payment against loan principal plus interest.

    Given:
     * a present value, `pv` (e.g., an amount borrowed)
     * a future value, `fv` (e.g., 0)
     * an interest `rate` compounded once per period, of which
       there are
     * `nper` total
     * and (optional) specification of whether payment is made
       at the beginning (`when` = {'begin', 1}) or the end
       (`when` = {'end', 0}) of each period

    Return:
       the (fixed) periodic payment.

    Parameters
    ----------
    rate : array_like
        Rate of interest (per period)
    nper : array_like
        Number of compounding periods
    pv : array_like
        Present value
    fv : array_like,  optional
        Future value (default = 0)
    when : {{'begin', 1}, {'end', 0}}, {string, int}
        When payments are due ('begin' (1) or 'end' (0))

    Returns
    -------
    out : ndarray
        Payment against loan plus interest.  If all input is scalar, returns a
        scalar float.  If any input is array_like, returns payment for each
        input element. If multiple inputs are array_like, they all must have
        the same shape.

    Notes
    -----
    The payment is computed by solving the equation::

     fv +
     pv*(1 + rate)**nper +
     pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0

    or, when ``rate == 0``::

      fv + pv + pmt * nper == 0

    for ``pmt``.

    Note that computing a monthly mortgage payment is only
    one use for this function.  For example, pmt returns the
    periodic deposit one must make to achieve a specified
    future balance given an initial deposit, a fixed,
    periodically compounded interest rate, and the total
    number of periods.

    References
    ----------
    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).
       Open Document Format for Office Applications (OpenDocument)v1.2,
       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,
       Pre-Draft 12. Organization for the Advancement of Structured Information
       Standards (OASIS). Billerica, MA, USA. [ODT Document].
       Available:
       http://www.oasis-open.org/committees/documents.php
       ?wg_abbrev=office-formulaOpenDocument-formula-20090508.odt

    Examples
    --------
    >>> import numpy_financial as npf

    What is the monthly payment needed to pay off a $200,000 loan in 15
    years at an annual interest rate of 7.5%?

    >>> npf.pmt(0.075/12, 12*15, 200000)
    -1854.0247200054619

    In order to pay-off (i.e., have a future-value of 0) the $200,000 obtained
    today, a monthly payment of $1,854.02 would be required.  Note that this
    example illustrates usage of `fv` having a default value of 0.

    """
    when = _convert_when(when)
    (rate, nper, pv, fv, when) = map(np.array, [rate, nper, pv, fv, when])
    temp = (1 + rate)**nper
    mask = (rate == 0)
    masked_rate = np.where(mask, 1, rate)
    fact = np.where(mask != 0, nper,
                    (1 + masked_rate*when)*(temp - 1)/masked_rate)
    return -(fv + pv*temp) / fact

def npf_ipmt(rate, per, nper, pv, fv=0, when='end'):
    """
    Compute the interest portion of a payment.

    Parameters
    ----------
    rate : scalar or array_like of shape(M, )
        Rate of interest as decimal (not per cent) per period
    per : scalar or array_like of shape(M, )
        Interest paid against the loan changes during the life or the loan.
        The `per` is the payment period to calculate the interest amount.
    nper : scalar or array_like of shape(M, )
        Number of compounding periods
    pv : scalar or array_like of shape(M, )
        Present value
    fv : scalar or array_like of shape(M, ), optional
        Future value
    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
        When payments are due ('begin' (1) or 'end' (0)).
        Defaults to {'end', 0}.

    Returns
    -------
    out : ndarray
        Interest portion of payment.  If all input is scalar, returns a scalar
        float.  If any input is array_like, returns interest payment for each
        input element. If multiple inputs are array_like, they all must have
        the same shape.

    See Also
    --------
    ppmt, pmt, pv

    Notes
    -----
    The total payment is made up of payment against principal plus interest.

    ``pmt = ppmt + ipmt``

    Examples
    --------
    >>> import numpy as np
    >>> import numpy_financial as npf

    What is the amortization schedule for a 1 year loan of $2500 at
    8.24% interest per year compounded monthly?

    >>> principal = 2500.00

    The 'per' variable represents the periods of the loan.  Remember that
    financial equations start the period count at 1!

    >>> per = np.arange(1*12) + 1
    >>> ipmt = npf.ipmt(0.0824/12, per, 1*12, principal)
    >>> ppmt = npf.ppmt(0.0824/12, per, 1*12, principal)

    Each element of the sum of the 'ipmt' and 'ppmt' arrays should equal
    'pmt'.

    >>> pmt = npf.pmt(0.0824/12, 1*12, principal)
    >>> np.allclose(ipmt + ppmt, pmt)
    True

    >>> fmt = '{0:2d} {1:8.2f} {2:8.2f} {3:8.2f}'
    >>> for payment in per:
    ...     index = payment - 1
    ...     principal = principal + ppmt[index]
    ...     print(fmt.format(payment, ppmt[index], ipmt[index], principal))
     1  -200.58   -17.17  2299.42
     2  -201.96   -15.79  2097.46
     3  -203.35   -14.40  1894.11
     4  -204.74   -13.01  1689.37
     5  -206.15   -11.60  1483.22
     6  -207.56   -10.18  1275.66
     7  -208.99    -8.76  1066.67
     8  -210.42    -7.32   856.25
     9  -211.87    -5.88   644.38
    10  -213.32    -4.42   431.05
    11  -214.79    -2.96   216.26
    12  -216.26    -1.49    -0.00

    >>> interestpd = np.sum(ipmt)
    >>> np.round(interestpd, 2)
    -112.98

    """
    when = _convert_when(when)
    rate, per, nper, pv, fv, when = np.broadcast_arrays(rate, per, nper,
                                                        pv, fv, when)
    total_pmt = npf_pmt(rate, nper, pv, fv, when)
    ipmt = _rbl(rate, per, total_pmt, pv, when)*rate
    try:
        ipmt = np.where(when == 1, ipmt/(1 + rate), ipmt)
        ipmt = np.where(np.logical_and(when == 1, per == 1), 0, ipmt)
    except IndexError:
        pass
    return ipmt

def npf_ppmt(rate, per, nper, pv, fv=0, when='end'):
    """
    Compute the payment against loan principal.

    Parameters
    ----------
    rate : array_like
        Rate of interest (per period)
    per : array_like, int
        Amount paid against the loan changes.  The `per` is the period of
        interest.
    nper : array_like
        Number of compounding periods
    pv : array_like
        Present value
    fv : array_like, optional
        Future value
    when : {{'begin', 1}, {'end', 0}}, {string, int}
        When payments are due ('begin' (1) or 'end' (0))

    See Also
    --------
    pmt, pv, ipmt

    """
    total = npf_pmt(rate, nper, pv, fv, when)
    return total - npf_ipmt(rate, per, nper, pv, fv, when)

def _rbl(rate, per, pmt, pv, when):
    """
    This function is here to simply have a different name for the 'fv'
    function to not interfere with the 'fv' keyword argument within the 'ipmt'
    function.  It is the 'remaining balance on loan' which might be useful as
    it's own function, but is easily calculated with the 'fv' function.
    """
    return fv(rate, (per - 1), pmt, pv, when)

def fv(rate, nper, pmt, pv, when='end'):
    """
    Compute the future value.

    Given:
     * a present value, `pv`
     * an interest `rate` compounded once per period, of which
       there are
     * `nper` total
     * a (fixed) payment, `pmt`, paid either
     * at the beginning (`when` = {'begin', 1}) or the end
       (`when` = {'end', 0}) of each period

    Return:
       the value at the end of the `nper` periods

    Parameters
    ----------
    rate : scalar or array_like of shape(M, )
        Rate of interest as decimal (not per cent) per period
    nper : scalar or array_like of shape(M, )
        Number of compounding periods
    pmt : scalar or array_like of shape(M, )
        Payment
    pv : scalar or array_like of shape(M, )
        Present value
    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional
        When payments are due ('begin' (1) or 'end' (0)).
        Defaults to {'end', 0}.

    Returns
    -------
    out : ndarray
        Future values.  If all input is scalar, returns a scalar float.  If
        any input is array_like, returns future values for each input element.
        If multiple inputs are array_like, they all must have the same shape.

    Notes
    -----
    The future value is computed by solving the equation::

     fv +
     pv*(1+rate)**nper +
     pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0

    or, when ``rate == 0``::

     fv + pv + pmt * nper == 0

    References
    ----------
    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).
       Open Document Format for Office Applications (OpenDocument)v1.2,
       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,
       Pre-Draft 12. Organization for the Advancement of Structured Information
       Standards (OASIS). Billerica, MA, USA. [ODT Document].
       Available:
       http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula
       OpenDocument-formula-20090508.odt

    Examples
    --------
    >>> import numpy as np
    >>> import numpy_financial as npf

    What is the future value after 10 years of saving $100 now, with
    an additional monthly savings of $100.  Assume the interest rate is
    5% (annually) compounded monthly?

    >>> npf.fv(0.05/12, 10*12, -100, -100)
    15692.928894335748

    By convention, the negative sign represents cash flow out (i.e. money not
    available today).  Thus, saving $100 a month at 5% annual interest leads
    to $15,692.93 available to spend in 10 years.

    If any input is array_like, returns an array of equal shape.  Let's
    compare different interest rates from the example above.

    >>> a = np.array((0.05, 0.06, 0.07))/12
    >>> npf.fv(a, 10*12, -100, -100)
    array([ 15692.92889434,  16569.87435405,  17509.44688102]) # may vary

    """
    when = _convert_when(when)
    (rate, nper, pmt, pv, when) = map(np.asarray, [rate, nper, pmt, pv, when])
    temp = (1+rate)**nper
    fact = np.where(rate == 0, nper,
                    (1 + rate*when)*(temp - 1)/rate)
    return -(pv*temp + pmt*fact)
###
###
###


def get_madad(initial_value=0.025,
              initial_steady_period_months_duration=12,
              steady_ramp_months_duration=84,
              long_range_centerline=0.03):

    madad_0 = initial_value * np.ones(initial_steady_period_months_duration)
    madad_1 = np.polyval([(long_range_centerline - initial_value) / steady_ramp_months_duration, madad_0[-1]],
                         np.arange(steady_ramp_months_duration))
    madad_3 = madad_1[-1] * np.ones(MAX_DURATION)
    madad = np.concatenate([madad_0, madad_1, madad_3])
    return madad


def get_prime(initial_value=0.06,
              initial_steady_period_months_duration=12,
              steady_ramp_months_duration=84,
              long_range_centerline=0.035,
              banks_margine=0.015,
              wavewlwngthy=10,
              amplitude=0.015):
    try:
        initial_value = requests.get('https://Boi.org.il/PublicApi/GetInterest').json()['currentInterest'] / 100 + 0.015
    except:
        pass
    prime_0 = initial_value * np.ones(initial_steady_period_months_duration)
    prime_1 = np.polyval([long_range_centerline / steady_ramp_months_duration, prime_0[-1]],
                         np.arange(steady_ramp_months_duration))
    prime_2 = np.sin(2 * np.pi * np.arange(0, 30, 1 / 12) / wavewlwngthy) * amplitude + prime_1[-1]
    prime = np.concatenate([prime_0, prime_1, prime_2]) + banks_margine
    return prime


MADAD = get_madad()
PRIME = get_prime()


def yearly_rate_to_monthly(yearly_rate):
    return yearly_rate / 12


def nominal_interest_rate_to_effective(nominal_rate):
    return (1 + yearly_rate_to_monthly(nominal_rate)) ** 12 - 1


def get_shpitzer_amortization(monthly_rate, duration, principal, return_dict=False, return_lists=False):
    try:
        len(monthly_rate)
    except TypeError:
        monthly_rate = monthly_rate * np.ones(duration)
    pmt = npf_pmt(monthly_rate[:duration], duration, principal, fv=0,
                  when=0)  # monthly payment against loan principal plus interest
    per = np.arange(duration) + 1  # periods (months)
    ipmt = npf_ipmt(monthly_rate[:duration], per, duration, principal)  # interest portion of a payment
    ppmt = npf_ppmt(monthly_rate[:duration], per, duration, principal)  # payment against loan principal
    if return_lists:
        d = {
            'pmt': pmt.tolist(),
            # 'per': per.tolist(),
            'ipmt': ipmt.tolist(),
            'ppmt': ppmt.tolist()
        }
    else:
        d = {
            'pmt': pmt,
            # 'per': per,
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
            # 'per': per.tolist(),
            'ipmt': ipmt.tolist(),
            'ppmt': ppmt.tolist()
        }
    else:
        d = {
            'pmt': pmt,
            # 'per': per,
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
                                                                              1,
                                                                              return_dict=return_dict,
                                                                              return_lists=return_lists)
            if equal:
                bank['equal'][title][duration] = get_equal_amortization(monthly_rate,
                                                                        duration,
                                                                        1,
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
    # bundles = list(sorted(bundles, key=lambda bundle: max(map(lambda b: b[-1], bundle))))
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
    func = lambda X: amortization_target_function(X,
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
            f_val = func((x1, x2, x3))
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
                                        tol=0.05,
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
        constraints = [
            NonlinearConstraint(lambda x: principal - x.sum(), 0, max_prime_content * principal),
            # NonlinearConstraint(lambda x: x.sum(), principal, principal),
        ]

        result = differential_evolution(func,
                                        bounds,
                                        constraints=constraints,
                                        disp=False,
                                        # maxiter=10,
                                        atol=1e3,
                                        tol=0.05,
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
                      madad_added_yearly_rate=0.01,
                      prime_added_yearly_rate=0.01,
                      increments=CALC_INCREMENTS,
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
    return best_bundle, rates


def sleek_optimize_mortgage(loan_principal,
                            max_monthly_payment,
                            amortizations,
                            use_max_prime=True,
                            fixed_yearly_rate=0.035,
                            madad_added_yearly_rate=0.01,
                            prime_added_yearly_rate=0.01,
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
        def func(X, for_opt=True):
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
        result = differential_evolution(func,
                                        bounds,
                                        # constraints=constraints,
                                        disp=False,
                                        # maxiter=10,
                                        atol=1e3,
                                        tol=0.05,
                                        popsize=9,
                                        seed=SEED,
                                        workers=1)
        return func(result.x, for_opt=False), rates


def bundle_to_df(bundle,
                 fixed_yearly_rate=0.035,
                 madad_added_yearly_rate=0.01,
                 prime_added_yearly_rate=0.01):
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


def MAIN(asset_cost,
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
        prime_added_yearly_rate = 0.00011 #-0.0064 * 1.1
    else:
        fixed_yearly_rate = lambda r: interp1d([0., .45, .6, .7, .75], [.03, .031, .033, .035, .035], kind='cubic')(r)
        madad_added_yearly_rate = lambda r: interp1d([0., .45, .6, .7, .75], [.0195, .0205, .0225, .0245, .0245], kind='cubic')(r)
        prime_added_yearly_rate = 0.0001 #-0.0064

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
                      madad_added_yearly_rate=madad_added_yearly_rate(funding_rate),
                      prime_added_yearly_rate=prime_added_yearly_rate)

    # else:
    #     raise ValueError(
    #         f'at least one method of amortization should be considered (Shpitzer or, and Equal)')
    #
    return df

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
    footer += f'<h6> {":כל הזכויות של&nbsp;אופטימייזר המשכנתא&nbsp; שמורות ליוצר"} </h6>'
    footer += f'<h6><a href="http://linkedin.com/in/natanel-davidovits-28695312" target="_blank" rel="noopener"> {"נתנאל דוידוביץ"} </a></h6>\n'
    footer += '''
            </body>
            </html>
        '''
    df = header + df + footer
    return df

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

    # asset_cost = 2000000
    # capital = 1000000
    # max_monthly_payment = 6000
    # net_monthly_income = 26000
    # is_married_couple = True
    # is_single_asset = True
    # amortizations = ['shpitzer']  # , 'equal']
    # prime = 1

    # s = f'''
    # {asset_cost}, {type(asset_cost)}
    # {capital}, {type(capital)}
    # {net_monthly_income}, {type(net_monthly_income)}
    # {max_monthly_payment}, {type(max_monthly_payment)}
    # {amortizations}, {type(amortizations)}
    # {is_married_couple}, {type(is_married_couple)}
    # {is_single_asset}, {type(is_single_asset)}
    # {prime}, {type(prime)}
    # '''

    df = MAIN(asset_cost,
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

    # output_div = document.querySelector("#output")
    # output_div.innerText = html_out #df
    # output_div.contentFrame = html_out
    new = window.open()
    new.document.body.innerHTML = html_out





if __name__ == '__main__':
    # func(0)
    ...
    # fixed_yearly_rate = 0.035
    # madad_added_yearly_rate = 0.035
    # prime_added_yearly_rate = -0.008
    #
    # # monthly_rate = yearly_rate_to_monthly(fixed_yearly_rate)
    # madad_monthly_rate = yearly_rate_to_monthly(MADAD + madad_added_yearly_rate)
    # prime_monthly_rate = yearly_rate_to_monthly(PRIME + prime_added_yearly_rate)
    #
    # tic = time()
    # bank = build_base_amortization_bank_dictionary(monthly_rate,
    #                                                madad_monthly_rate,
    #                                                prime_monthly_rate,
    #                                                return_dict=False,
    #                                                return_lists=False)
    # print(time() - tic)
    # # with open('bank.json', 'w') as f:
    # #     json.dump(bank, f)
    #
    # # tic = time()
    # # with open('bank.json', 'r') as f:
    # #     bank = json.load(f)
    # # print(time() - tic)
    #
    #
    # principals = {'shpitzer': {'fixed': 3e4, 'madad': 3e4, 'prime': 3e4},
    #               'equal': {'fixed': 3e4, 'madad': 3e4, 'prime': 3e4}}
    # max_monthly_payment = 1e3
    # print('----')
    #
    # tic = time()
    # _bank = principize_base_bank(principals, bank)
    # print(time() - tic)
    # for a in _bank.keys():
    #     for r in _bank[a].keys():
    #         print(a, r, len(_bank[a][r]))#, list(_bank[a][r].keys()))
    # # pp(_bank)
    # # print('----')
    #
    # tic = time()
    # shpitzer_bundle = amortized_bundle_filter_by_max_monthly_payment(_bank, 'shpitzer', max_monthly_payment)
    # # equal_bundle = amortized_bundle_filter_by_max_monthly_payment(_bank, 'equal', max_monthly_payment)
    # print(time() - tic)
    # # print(len(shpitzer_bundle), len(equal_bundle))
    #
    # tic = time()
    # opt = optimize_bundles_filter(_bank, [shpitzer_bundle], max_monthly_payment)
    # print(time() - tic)
    # # print(opt)

    # loan_principal = 1200000
    # max_monthly_payment = 6327
    # amortizations = ['shpitzer', 'equal']
    # tic = time()
    # best_bundle = optimize_mortgage(loan_principal,
    #                                 max_monthly_payment,
    #                                 amortizations,
    #                                 use_max_prime=False)
    # # best_bundle = sleek_optimize_mortgage(loan_principal,
    # #                                         max_monthly_payment,
    # #                                         amortizations,
    # #                                         use_max_prime=False)
    #
    # print(time() - tic)
    #
    # print('---')
    # print(best_bundle.keys())
    # tot = 0
    # for k1 in best_bundle.keys():
    #     for k2 in best_bundle[k1].keys():
    #         for k3 in best_bundle[k1][k2].keys():
    #             amort_sum = best_bundle[k1][k2][k3]['ppmt']
    #             print(k1, k2, k3, -amort_sum, -amort_sum / loan_principal)
    #             tot += amort_sum
    # print(tot)

    # # plotting:
    # plt.plot(madad_monthly_rate*100, label='מדד המחירים לצרכן'[::-1])
    # plt.plot(prime_monthly_rate*100, label='ריבית הפריים'[::-1])
    # plt.xlabel('חודשים'[::-1])
    # plt.ylabel('%')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()





    asset_cost = 2000000
    capital = 1000000
    max_monthly_payment = 6000
    net_monthly_income = 26000
    is_married_couple = True
    is_single_asset = True
    amortizations = ['shpitzer']  # , 'equal']
    prime = 1

    from time import time
    tic = time()
    df = MAIN(asset_cost,
                 capital,
                 max_monthly_payment,
                 net_monthly_income,
                 amortizations,
                 is_married_couple=is_married_couple,
                 is_single_asset=is_single_asset,
                 prime=prime)

    html_out = beutify_HTML(df, asset_cost,
                                 capital,
                                 max_monthly_payment,
                                 net_monthly_income,
                                 is_married_couple,
                                 is_single_asset,
                                 amortizations,
                                 prime)


    print(df)
    print(time() - tic)


