"""Reproduce the mathematical constructions in mathematical_checks.pdf.

Python 3.9+; standard library only. No datasets, training, or random draws.
ARM-FM and unstandardized gradients use exact rational arithmetic. Standardized
groups are exhaustively enumerated with exact multinomial probabilities and
floating-point square roots. This is review-owned analysis, not source results.

Usage: python reproduce_checks.py [--output results.json]
"""

import argparse
from fractions import Fraction as F
from itertools import product
import json
from math import comb, factorial, fsum, isclose, sqrt
from pathlib import Path


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def gradient(p, f):
    average = dot(p, f)
    return tuple(pi * (fi - average) for pi, fi in zip(p, f))


def exact(x):
    if isinstance(x, F):
        return {"fraction": str(x), "decimal": float(x)}
    if isinstance(x, (tuple, list)):
        return [exact(v) for v in x]
    if isinstance(x, dict):
        return {k: exact(v) for k, v in x.items()}
    return x


def check_close(x, y):
    assert isclose(float(x), float(y), rel_tol=1e-10, abs_tol=1e-12), (x, y)


def count_states(p, group_size):
    """All multinomial count vectors for the three-action policy."""
    total = F(0)
    states = []
    for n0 in range(group_size + 1):
        for n1 in range(group_size - n0 + 1):
            ns = (n0, n1, group_size - n0 - n1)
            multiplicity = factorial(group_size)
            for n in ns:
                multiplicity //= factorial(n)
            probability = F(multiplicity)
            for pi, n in zip(p, ns):
                probability *= pi ** n
            total += probability
            states.append((ns, probability))
    assert total == 1
    assert len(states) == comb(group_size + 2, 2)
    return states


def group_gradient(p, scores, group_size, *, epsilon=0., sample_std=False,
                   normalized_terms=1, centered_terms=0, save_states=False):
    """Initial expected one-token surrogate gradient, fixed sampled advantages.

    A_i = normalized_terms*(S_i-mean)/(std+epsilon)
          + centered_terms*(S_i-mean).
    All-equal groups receive zero advantage. This explicit convention also
    handles epsilon=0; absent actions make no contribution to a group.
    """
    terms = [[], [], []]
    saved = []
    for ns, probability in count_states(p, group_size):
        mean = sum(n * s for n, s in zip(ns, scores)) / group_size
        centered = tuple(s - mean for s in scores)
        var = sum(n * d * d for n, d in zip(ns, centered)) / group_size
        if sample_std:
            var *= F(group_size, group_size - 1)
        sigma = sqrt(float(var))
        normalized = ([float(d) / (sigma + epsilon) for d in centered]
                      if var else [0.] * 3)
        advantage = [normalized_terms * a + centered_terms * float(d)
                     for a, d in zip(normalized, centered)]
        g = [0.] * 3
        for i, n in enumerate(ns):
            for k in range(3):
                h = float(F(int(i == k)) - p[k])
                g[k] += n / group_size * advantage[i] * h
        for k in range(3):
            terms[k].append(float(probability) * g[k])
        if save_states:
            saved.append({"counts": list(ns), "probability": exact(probability),
                          "mean_score": exact(mean), "std": sigma,
                          "advantages_by_action": advantage,
                          "group_gradient": g})
    result = {"G": group_size, "count_states": comb(group_size + 2, 2),
              "epsilon": epsilon, "std": "sample" if sample_std else "population",
              "normalized_terms": normalized_terms, "centered_terms": centered_terms,
              "expected_gradient": [fsum(v) for v in terms]}
    check_close(sum(result["expected_gradient"]), 0)
    if save_states:
        result["states"] = saved
    return result


def centered_gradient_exact(p, scores, group_size, copies=1):
    """Independently enumerate the self-included mean baseline in fractions."""
    g = [F(0)] * 3
    for ns, probability in count_states(p, group_size):
        mean = sum(n * s for n, s in zip(ns, scores)) / group_size
        for i, n in enumerate(ns):
            advantage = copies * (scores[i] - mean)
            for k in range(3):
                g[k] += probability * F(n, group_size) * advantage * (int(i == k) - p[k])
    multiplier = F(copies * (group_size - 1), group_size)
    assert tuple(g) == tuple(multiplier * x for x in gradient(p, scores))
    return tuple(g)


def arm_check():
    gamma, bonus = F(99, 100), F(1, 5)
    direct = {"task": F(1), "machine": F(1), "additive": F(2)}
    delayed = {"task": gamma, "machine": bonus + gamma,
               "additive": bonus + 2 * gamma}
    assert 0 < bonus < 1 and delayed["task"] < direct["task"]
    assert delayed["additive"] > direct["additive"]
    assert delayed["machine"] > direct["machine"]
    assert bonus > 2 * (1 - gamma)
    return exact({"gamma": gamma, "intermediate_bonus": bonus,
                  "direct": direct, "delayed": delayed,
                  "RM_only_reversal_threshold": 1 - gamma,
                  "additive_reversal_threshold": 2 * (1 - gamma)})


def calibration_check():
    p = (F(1, 10), F(3, 5), F(3, 10))
    c = (F(1), F(1), F(0))
    r = (F(1), F(2, 3), F(2, 3))
    gc, gr = gradient(p, c), gradient(p, r)
    for score in set(r):
        ids = [i for i in range(3) if r[i] == score]
        assert sum(p[i] * c[i] for i in ids) / sum(p[i] for i in ids) == score
    assert dot(p, c) == dot(p, r) == F(7, 10)
    assert dot(gc, gr) == -F(6, 10000)
    assert dot(gr, gr) == F(14, 10000)
    # Compute conditional covariance directly, using h_i=e_i-p.
    covariance = [F(0)] * 3
    for score in set(r):
        ids = [i for i in range(3) if r[i] == score]
        mass = sum(p[i] for i in ids)
        hbar = [sum(p[i] * (int(i == k) - p[k]) for i in ids) / mass
                for k in range(3)]
        for k in range(3):
            covariance[k] += sum(p[i] * (c[i] - score)
                                 * ((int(i == k) - p[k]) - hbar[k]) for i in ids)
    assert tuple(gr[k] - gc[k] for k in range(3)) == tuple(-v for v in covariance)
    groups = {}
    for G in (2, 8, 16):
        kappa = fsum(comb(G, n) * float(p[0] ** n * (1 - p[0]) ** (G - n))
                     * sqrt(n / G * (1 - n / G)) for n in range(G + 1))
        enumerated = group_gradient(p, r, G)
        expected = [kappa, -2 * kappa / 3, -kappa / 3]
        for x, y in zip(enumerated["expected_gradient"], expected):
            check_close(x, y)
        derivative = dot([float(x) for x in gc], expected)
        check_close(derivative, -.02 * kappa)
        no_std = centered_gradient_exact(p, r, G)
        assert dot(gc, no_std) == F(G - 1, G) * dot(gc, gr) < 0
        groups[str(G)] = {"kappa": kappa, "expected_gradient": expected,
                          "success_derivative": derivative,
                          "no_std_gradient": exact(no_std),
                          "no_std_success_derivative": exact(dot(gc, no_std)),
                          "count_states": enumerated["count_states"]}
    ideal_adam = (F(1), F(-1), F(-1))
    assert dot(gc, ideal_adam) == F(6, 100)
    tilt = dot(p, tuple(x * x for x in r)) - dot(p, r) ** 2
    assert tilt == F(1, 100)
    return exact({"p": p, "c": c, "r": r, "success_gradient": gc,
                  "score_gradient": gr, "score_dot_score": dot(gr, gr),
                  "success_dot_score": dot(gc, gr), "conditional_covariance": covariance,
                  "GSPO_groups": groups, "ideal_full_gradient_Adam_success_derivative": dot(gc, ideal_adam),
                  "reward_tilt_initial_success_derivative": tilt})


def procredit_check():
    p = (F(2, 100), F(75, 100), F(23, 100))
    c, phi = (F(1), F(0), F(0)), (F(1), F(1, 10), F(0))
    b = F(1, 2)
    s = tuple(ci + b * ph for ci, ph in zip(c, phi))
    gc, gs = gradient(p, c), gradient(p, s)
    # Exact ordered-pair enumeration, independently checked by count states.
    g, gt, gu = [F(0)] * 3, [F(0)] * 3, [F(0)] * 3
    ordered_pairs = []
    for i, j in product(range(3), repeat=2):
        delta = s[i] - s[j]
        sign = F((delta > 0) - (delta < 0))
        A = sign + delta / 2
        local = [F(0)] * 3
        local[i] += A / 2
        local[j] -= A / 2
        probability = p[i] * p[j]
        for k in range(3):
            g[k] += probability * local[k]
        gt[i] += probability * sign / 2
        gt[j] -= probability * sign / 2
        gu[i] += probability * delta / 4
        gu[j] -= probability * delta / 4
        ordered_pairs.append(exact({"actions": [i + 1, j + 1], "probability": probability,
                                    "advantages": [A, -A], "group_gradient": local}))
    assert tuple(gu) == tuple(x / 2 for x in gs)
    assert dot(gc, gs) == F(".00082983")
    assert dot(gc, g) == F("-.000748765")
    assert dot(gc, gt) == F("-.00116368")
    assert dot(gc, gu) == F(".000414915")
    assert dot(gs, g) > 0
    check_g2 = group_gradient(p, s, 2, centered_terms=1)["expected_gradient"]
    for x, y in zip(check_g2, g):
        check_close(x, y)
    finite = {}
    for G in (2, 8):
        for sample_std in (False, True):
            result = group_gradient(p, s, G, epsilon=1e-6, sample_std=sample_std,
                                    centered_terms=1, save_states=(G == 8))
            result["success_derivative"] = dot([float(x) for x in gc], result["expected_gradient"])
            result["score_derivative"] = dot([float(x) for x in gs], result["expected_gradient"])
            assert result["success_derivative"] < 0 < result["score_derivative"]
            finite[f"G{G}_{result['std']}"] = result
    check_close(finite["G8_population"]["success_derivative"], -.000929983747842)
    check_close(finite["G8_sample"]["success_derivative"], -.000823033652568)
    no_std = {}
    for G in (2, 8):
        result = centered_gradient_exact(p, s, G, copies=2)
        assert dot(gc, result) > 0
        no_std[str(G)] = exact({"gradient": result, "success_derivative": dot(gc, result),
                               "raw_score_multiplier": F(2 * (G - 1), G)})
    sigma = sqrt(float(dot(p, tuple(x * x for x in s)) - dot(p, s) ** 2))
    large_G_derivative = (1 / (sigma + 1e-6) + 1) * float(dot(gc, gs))
    assert large_G_derivative > 0
    tiny_s = tuple(ci + F(1, 10**9) * ph for ci, ph in zip(c, phi))
    tiny_g = group_gradient(p, tiny_s, 2, epsilon=1e-6, centered_terms=1)["expected_gradient"]
    tiny_derivative = dot([float(x) for x in gc], tiny_g)
    assert tiny_derivative > 0
    return exact({"p": p, "c": c, "final_progress": phi, "b": b, "scores": s,
                  "success_gradient": gc, "raw_score_gradient": gs,
                  "G2_exact": {"ordered_pairs": ordered_pairs, "combined_gradient": g,
                               "trajectory_gradient": gt, "turn_gradient": gu,
                               "success_dot_raw": dot(gc, gs), "success_dot_combined": dot(gc, g),
                               "success_dot_trajectory": dot(gc, gt), "success_dot_turn": dot(gc, gu),
                               "score_dot_combined": dot(gs, g)},
                  "finite_groups_epsilon_1e_6": finite, "no_standardization": no_std,
                  "large_group_success_derivative_epsilon_1e_6": large_G_derivative,
                  "tiny_b_1e_9_G2_success_derivative_epsilon_1e_6": tiny_derivative})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    result = {"kind": "Analytical constructions, not original-paper training results",
              "ARM_FM": arm_check(), "SWE_RM": calibration_check(),
              "ProCredit": procredit_check(),
              "all_checks_passed": True}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PASS: discounted return, calibration, GSPO, and ProCredit checks.")
    print("Exact multinomial probabilities; no training or Monte Carlo sampling.")
    print("Results:", args.output)


if __name__ == "__main__":
    main()
