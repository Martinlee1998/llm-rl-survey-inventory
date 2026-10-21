"""Independent checks of the NEW calibrated three-action family.

Standard library only. Does not import/run the existing reproduce_checks.py.
For each base policy p, r is frozen before taking ambient softmax-logit
 derivatives. Exact rational identities, seeded random sign checks, a central
finite-difference check, and independent multinomial GSPO enumeration are used.
No training, no ProCredit recomputation, and no frequency/prevalence estimate.
"""

import argparse
from fractions import Fraction as F
from itertools import product
import json
from math import comb, exp, factorial, fsum, isclose, log, sqrt
from pathlib import Path
from random import Random


def dot(x, y):
    return sum(xi * yi for xi, yi in zip(x, y))


def grad(p, f):
    jf = dot(p, f)
    return tuple(pi * (fi - jf) for pi, fi in zip(p, f))


def family(a, q):
    m = 1 - a
    return (a, m * q, m * (1 - q)), (F(1), q, q)


def h(a, q):
    return 2 * a * (q*q - q + 1) - q * (2*q - 1)


def threshold_a(q):
    return q * (2*q - 1) / (2 * (q*q - q + 1))


def bracket_ad(a, d):
    return (1-a)*(2*a-1) + (3-2*a)*d - 2*d*d


def threshold_d(a):
    return ((3-2*a) - sqrt(1+12*a*(1-a))) / 4


def softmax(z):
    top = max(z)
    es = [exp(zi-top) for zi in z]
    return [ei / fsum(es) for ei in es]


def close(x, y, tol=2e-12):
    assert isclose(float(x), float(y), rel_tol=tol, abs_tol=tol), (x, y)


def encode(x):
    if isinstance(x, F):
        return {'exact': str(x), 'float': float(x)}
    if isinstance(x, dict):
        return {str(k): encode(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [encode(v) for v in x]
    return x


def rational_identities(rng):
    c = (F(1), F(1), F(0))
    max_poly_error = F(0)
    for _ in range(1200):
        n = [rng.randint(1, 200) for _ in range(3)]
        p = tuple(F(k, sum(n)) for k in n)
        a, b, d = p
        m, q = 1-a, b/(b+d)
        r = (F(1), q, q)
        gc, gr = grad(p, c), grad(p, r)
        assert dot(p, c) == dot(p, r) == 1-d
        assert b/(b+d) == q  # exact low-score calibration; high score is correct
        assert gc == (a*d, b*d, -d*(1-d))
        assert gr == (a*d, -a*b*d/m, -a*d*d/m)
        lhs = dot(gc, gr)
        rhs_ad = a*d*d/m * bracket_ad(a, d)
        rhs_q = a*m*m*(1-q)**2 * h(a, q)
        assert lhs == rhs_ad == rhs_q
        assert bracket_ad(a, d) == a*m - b*b + d*(1-d)
        assert bracket_ad(a, d) == m*h(a, q)
        assert dot(gr, gr) > 0
        assert (lhs < 0) == (q > F(1, 2) and a < threshold_a(q))
        # The calibration curve with q held fixed is a different derivative.
        dz_da = (1/a, -1/m, -1/m)
        assert dot(gc, dz_da) == dot(gr, dz_da) == 1-q
        # Exponential tilt / Fisher natural-gradient velocity has a good sign.
        variance = dot(p, tuple(ri*ri for ri in r)) - dot(p, r)**2
        assert variance == a*m*(1-q)**2 == a*d*d/m > 0
        assert dot(gc, (F(1), F(-1), F(-1))) == 2*a*d > 0
        # Exact boundary H(A(q),q)=0, including q <= 1/2 algebraically.
        assert h(threshold_a(q), q) == 0
        max_poly_error = max(max_poly_error, abs(lhs-rhs_q))
    return {'points': 1200, 'max_exact_identity_error': max_poly_error}


def fixed_evaluator_slice():
    q, c = F(2, 3), (F(1), F(1), F(0))
    examples = []
    for a in [F(1, 100), F(1, 10), F(1, 7), F(1, 5), F(4, 5)]:
        p, r = family(a, q)
        gc, gr = grad(p, c), grad(p, r)
        expected = 2*a*(1-a)**2*(7*a-1)/81
        assert dot(gc, gr) == expected
        assert dot(p, c) == dot(p, r) == (2+a)/3
        examples.append({'a': a, 'p': p, 'r': r, 'gc_dot_gr': expected})
    # Exact radical root for a=1/10: d_*=(7-sqrt(13))/10.
    # Write d_*=x+y*sqrt(13) and reduce the quadratic symbolically.
    x, y = F(7, 10), F(-1, 10)
    rational_part = -2*(x*x + 13*y*y) + F(14, 5)*x - F(18, 25)
    radical_part = -4*x*y + F(14, 5)*y
    assert rational_part == radical_part == 0
    root = (7-sqrt(13))/10
    close(root, threshold_d(.1))
    # A genuinely open rectangle around the original point in (a,d).
    # Both partial derivatives of B are positive throughout this rectangle;
    # its supremum is B(.11,.31) = -123/5000 < 0.
    upper = bracket_ad(F(11, 100), F(31, 100))
    min_da = 3 - 4*F(11, 100) - 2*F(31, 100)
    min_dd = 3 - 2*F(11, 100) - 4*F(31, 100)
    assert upper == -F(123, 5000) and min_da > 0 and min_dd > 0
    return {'fixed_q': q, 'threshold_a': threshold_a(q), 'examples': examples,
            'a_0_1_d_root_exact': '(7-sqrt(13))/10', 'a_0_1_d_root': root,
            'root_polynomial_coefficients': [rational_part, radical_part],
            'open_rectangle_a': ['9/100', '11/100'],
            'open_rectangle_d': ['29/100', '31/100'],
            'bracket_supremum_on_rectangle': upper,
            'rectangle_coordinate_area': F(1, 2500)}


def floating_sign_and_derivative_checks(rng):
    c = (1., 1., 0.)
    max_c_error = max_r_error = 0.
    for k in range(6000):
        a = rng.uniform(.002, .998)
        d = (1-a)*rng.uniform(.002, .998)
        b = 1-a-d
        q = b/(1-a)
        p, r = (a, b, d), (1., q, q)
        gc, gr = grad(p, c), grad(p, r)
        z = [log(pi) for pi in p]
        sign_from_d = 0 < a < .5 and 0 < d < threshold_d(a)
        sign_from_q = q > .5 and a < threshold_a(q)
        assert sign_from_d == sign_from_q == (dot(gc, gr) < 0)
        if k < 80:
            step = 1e-4
            pplus = softmax([zi+step*gi for zi, gi in zip(z, gr)])
            pminus = softmax([zi-step*gi for zi, gi in zip(z, gr)])
            dc = (dot(pplus,c)-dot(pminus,c))/(2*step)
            dr = (dot(pplus,r)-dot(pminus,r))/(2*step)
            ec, er = abs(dc-dot(gc,gr)), abs(dr-dot(gr,gr))
            assert ec < 2e-9 and er < 2e-9
            max_c_error, max_r_error = max(max_c_error,ec), max(max_r_error,er)
    return {'random_sign_points': 6000, 'finite_difference_points': 80,
            'finite_difference_step': 1e-4,
            'max_success_derivative_error': max_c_error,
            'max_score_derivative_error': max_r_error}


def count_gradient(p, r, G, epsilon, sample_std):
    """Direct expected score-function sum over every three-action count vector.

    This implementation uses individual centered advantages and e_i-p directly;
    it does not assume the cancellation or collinearity being checked.
    """
    assert G >= 2 and epsilon > 0
    weights = F(0)
    terms = [[], [], []]
    centered = [F(0), F(0), F(0)]
    for n0 in range(G+1):
        for n1 in range(G-n0+1):
            counts = (n0,n1,G-n0-n1)
            mult = factorial(G)
            for ni in counts:
                mult //= factorial(ni)
            prob = F(mult)
            for pi, ni in zip(p,counts):
                prob *= pi**ni
            weights += prob
            mean = sum(F(ni,G)*ri for ni,ri in zip(counts,r))
            variance = sum(F(ni,G)*(ri-mean)**2 for ni,ri in zip(counts,r))
            if sample_std:
                variance *= F(G,G-1)
            denom = sqrt(float(variance)) + epsilon
            for j in range(3):
                local = 0.
                exact_local = F(0)
                for i in range(3):
                    factor = F(counts[i],G)*(int(i==j)-p[j])
                    local += float(factor)*float(r[i]-mean)/denom
                    exact_local += factor*(r[i]-mean)
                terms[j].append(float(prob)*local)
                centered[j] += prob*exact_local
    assert weights == 1
    return tuple(fsum(ts) for ts in terms), tuple(centered)


def scalar_kappa(a,q,G,epsilon,sample_std):
    delta = float(1-q)
    scale = sqrt(G/(G-1)) if sample_std else 1.
    return fsum(float(F(comb(G,n))*a**n*(1-a)**(G-n))
                * delta*(n/G)*(1-n/G)
                / (scale*delta*sqrt((n/G)*(1-n/G))+epsilon)
                for n in range(G+1))


def gspo_family_checks(rng):
    points = [(F(1,10),F(2,3)),(F(1,7),F(2,3)),
              (F(1,5),F(2,3)),(F(1,20),F(4,5)),
              (F(1,5),F(1,3))]
    for _ in range(3):
        points.append((F(rng.randint(1,19),20),F(rng.randint(1,19),20)))
    cases, max_error, summaries = 0, 0., []
    c = (F(1),F(1),F(0))
    for a,q in points:
        p,r = family(a,q)
        gc,gr = grad(p,c),grad(p,r)
        for G in [2,3,8,16]:
            for eps in [1e-6,.05]:
                for use_sample in [False,True]:
                    observed,centered = count_gradient(p,r,G,eps,use_sample)
                    kappa = scalar_kappa(a,q,G,eps,use_sample)
                    expected = (kappa,-kappa*float(q),-kappa*float(1-q))
                    assert kappa > 0
                    for x,y in zip(observed,expected):
                        close(x,y)
                        max_error = max(max_error,abs(x-y))
                    raw_multiplier = kappa/float(a*(1-a)*(1-q))
                    for x,y in zip(observed,gr):
                        close(x,raw_multiplier*float(y))
                    assert centered == tuple(F(G-1,G)*gi for gi in gr)
                    expected_success = kappa*float((1-a)*(1-q)*h(a,q))
                    close(dot(gc,observed),expected_success)
                    cases += 1
                    if G==8 and eps==1e-6 and not use_sample:
                        summaries.append({'a':a,'q':q,'kappa':kappa,
                                          'gc_dot_expected_gspo':expected_success,
                                          'gc_dot_centered':dot(gc,centered)})
    return {'cases':cases,'G_values':[2,3,8,16],
            'epsilon_values':[1e-6,.05],
            'standard_deviation_conventions':['population','sample'],
            'max_gradient_identity_error':max_error,
            'G8_population_examples':summaries}


def coordinate_boundary():
    p,r=family(F(1,10),F(2,3))
    gc,gr=grad(p,(F(1),F(1),F(0))),grad(p,r)
    ordinary=dot(gc,gr)
    scaled=dot(gc,(2*gr[0],gr[1],gr[2]))
    assert ordinary == -F(3,5000)
    assert scaled == F(3,10000) > 0
    return {'ordinary_logit_step':ordinary,
            'D_diag_2_1_1_step':scaled,
            'ideal_first_adam_full_expected_gradient':dot(gc,(F(1),F(-1),F(-1))),
            'scope':'Coordinate scaling and ideal full-gradient Adam; not stochastic Adam.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,
                        default=Path(__file__).with_suffix('.results.json'))
    args=parser.parse_args()
    rng=Random(20261021)
    result={'scope':'New analytic family only; fixed scores for every local derivative',
            'seed':20261021,
            'exact_identities':rational_identities(rng),
            'fixed_evaluator_slice':fixed_evaluator_slice(),
            'independent_float_checks':floating_sign_and_derivative_checks(rng),
            'GSPO_family':gspo_family_checks(rng),
            'coordinate_boundary':coordinate_boundary(),
            'all_checks_passed':True}
    args.output.write_text(json.dumps(encode(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('PASS: fixed-evaluator family, general negative region, root, GSPO, coordinate boundary.')
    print('1200 exact rational points; 6000 seeded sign checks; 80 finite differences; 128 GSPO cases.')
    print('No existing reproduction script or ProCredit computation was run.')


if __name__=='__main__':
    main()
