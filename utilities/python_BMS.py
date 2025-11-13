#### New! Written by Kate ####
#### Bayesian Model Selection for Group Studies in Python ####
#### Converted from SPM_BMS.m https://github.com/neurodebian/spm12/blob/master/spm_BMS.m ####

import numpy as np
from scipy.special import psi, gammaln

def bms(lme, Nsamp=1_000_000, do_plot=False, sampling=False, ecp=True, alpha0=None):
    max_val = np.log(np.finfo(float).max)
    Ni, Nk = lme.shape
    c = 1
    cc = 1e-4

    if alpha0 is None:
        alpha0 = np.ones(Nk)
    alpha = alpha0.copy()

    while c > cc:
        log_u = np.zeros((Ni, Nk))
        for i in range(Ni):
            for k in range(Nk):
                log_u[i, k] = lme[i, k] + psi(alpha[k]) - psi(np.sum(alpha))
            log_u[i, :] -= np.mean(log_u[i, :])
            log_u[i, :] = np.sign(log_u[i, :]) * np.minimum(np.abs(log_u[i, :]), max_val)

        u = np.exp(log_u)
        u_i = np.sum(u, axis=1, keepdims=True)
        g = u / u_i

        beta = np.sum(g, axis=0)
        prev = alpha.copy()
        alpha = alpha0 + beta
        c = np.linalg.norm(alpha - prev)

    exp_r = alpha / np.sum(alpha)

    if ecp:
        if Nk == 2:
            xp = np.array([
                1 - beta_cdf(0.5, alpha[1], alpha[0]),
                1 - beta_cdf(0.5, alpha[0], alpha[1])
            ])
        else:
            xp = dirichlet_exceedance(alpha, Nsamp)
    else:
        xp = np.zeros(Nk)

    posterior = {'a': alpha, 'r': g.T}
    priors = {'a': alpha0}

    F1 = FE(lme.T, posterior, priors)
    F0 = FE_null(lme.T)
    bor = 1 / (1 + np.exp(F1 - F0))
    pxp = (1 - bor) * xp + bor / Nk

    return alpha, exp_r, xp, pxp, bor

def beta_cdf(x, a, b):
    from scipy.stats import beta
    return beta.cdf(x, a, b)

def dirichlet_exceedance(alpha, Nsamp):
    samples = np.random.dirichlet(alpha, Nsamp)
    winners = np.argmax(samples, axis=1)
    xp = np.bincount(winners, minlength=len(alpha)) / Nsamp
    return xp

def FE(L, posterior, priors):
    K, n = L.shape
    a0 = np.sum(posterior['a'])
    Elogr = psi(posterior['a']) - psi(np.sum(posterior['a']))

    Sqf = np.sum(gammaln(posterior['a'])) - gammaln(a0) - np.sum((posterior['a'] - 1) * Elogr)
    Sqm = -np.sum(posterior['r'] * np.log(posterior['r'] + np.finfo(float).eps))

    ELJ = gammaln(np.sum(priors['a'])) - np.sum(gammaln(priors['a'])) + np.sum((priors['a'] - 1) * Elogr)
    for i in range(n):
        for k in range(K):
            ELJ += posterior['r'][k, i] * (Elogr[k] + L[k, i])

    return ELJ + Sqf + Sqm

def FE_null(L):
    K, n = L.shape
    F0m = 0
    for i in range(n):
        tmp = L[:, i] - np.max(L[:, i])
        g = np.exp(tmp) / np.sum(np.exp(tmp))
        for k in range(K):
            F0m += g[k] * (L[k, i] - np.log(K) - np.log(g[k] + np.finfo(float).eps))
    return F0m
