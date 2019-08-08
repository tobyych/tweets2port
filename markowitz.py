import pandas as pd
import numpy as np
import cvxopt as opt
from cvxopt import solvers, blas
import matplotlib.pyplot as plt

solvers.options["show_progress"] = False


def optimal_port(returns, short_sell=False):
    returns = np.asmatrix(returns)

    n = len(returns)

    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    if short_sell:
        portfolios = [solvers.qp(mu * S, -pbar, A=A, b=b)["x"] for mu in mus]
        returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        wt = solvers.qp(opt.matrix(x1 * S), -pbar, A=A, b=b)["x"]
        return np.asarray(wt), returns, risks

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n, 1))

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G=G, h=h, A=A, b=b)["x"] for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G=G, h=h, A=A, b=b)["x"]
    return np.asarray(wt), returns, risks


def equally_weighted_port(returns):
    returns = np.asmatrix(returns)
    mean = np.mean(returns, axis=1)
    cov = np.cov(returns)
    n = len(returns)
    weights = np.ones(n) / n
    return (
        weights,
        np.dot(weights, mean).item(),
        np.sqrt(np.dot(weights, cov @ weights)),
    )


returns = pd.read_pickle("temp/pred.pickle")
_, opt_mean, opt_std = optimal_port(returns)
print(opt_mean, opt_std)
_, opt_ss_mean, opt_ss_std = optimal_port(returns, short_sell=True)
_, ew_mean, ew_std = equally_weighted_port(returns)
fig = plt.figure()
plt.plot(opt_std, opt_mean)
plt.plot(opt_ss_std, opt_ss_mean, "y")
plt.plot(ew_std, ew_mean, "ro", markersize=10)
plt.ylabel("mean")
plt.xlabel("std")
plt.xlim(0, 0.7)
plt.ylim(0, 1)
plt.show()
