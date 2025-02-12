#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:26:54 2025

@author: admin
"""

#imports
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Define parameters
n = 100000  # Length of the time series

# Generate time index
time = np.arange(n)

# Define a trend (Linear trend: y = a*t + b)
a = 0.0002  # Slope
b = .5   # Intercept
trend = a * time + b

# Optional: Add seasonality (sinusoidal pattern)
seasonality = 0#5 * np.sin(2 * np.pi * time / 20)  # Period = 20

# Add noise (Gaussian white noise)
noise = np.random.normal(0, 2, n)

# Create the process
process = trend + seasonality + noise

# Store in a DataFrame
df = pd.DataFrame({"time": time, "value": process})

# Plot the simulated time series
plt.figure(figsize=(12, 6),facecolor="white")
plt.plot(df["time"], df["value"], label="Simulated Process", color="blue", alpha=0.7)
plt.plot(df["time"], trend, label="Trend (Linear)", color="red", linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Simulated Time Series with Trend")
plt.legend()
plt.show()


class TVC_QAR:
    def __init__(self, vY, vU0, vU, fTau=0.5, h=0.1, kernel='Epanechnikov', p=1, n_jobs=-1):
        self.vY = vY.values.flatten()
        self.fTau = fTau
        self.h = h
        self.kernel = kernel
        self.p = p
        self.iT = len(vY)
        self.vU0 = vU0.flatten()
        self.U_t = vU.flatten()
        self.mAlpha = np.zeros((len(self.vU0), 2 * (p + 1)))
        self.qhat = np.zeros((len(self.vU0), self.iT - self.p))  # Store quantile predictions
        self.n_jobs = n_jobs
        self.X = self.makeX() if p > 0 else np.ones((self.iT, 1))
        self.a_n = (self.iT * self.h) ** -0.5  # Scaling factor

    def K(self, vU):
        """Define kernel function."""
        if self.kernel == 'Epanechnikov':
            return np.where(np.abs(vU) <= 1, 3 / 4 * (1 - vU ** 2), 0)
        elif self.kernel == 'Uniform':
            return np.where(np.abs(vU) <= 1, 1 / 2, 0)
        elif self.kernel == 'Normal':
            return norm.pdf(vU, 0, 1)
        else:
            raise NotImplementedError('Supported kernels: Epanechnikov, Uniform, Normal')

    def rho_tau(self, x):
        """Quantile loss function."""
        return x * (self.fTau - (x < 0))

    def makeX(self):
        """Create X matrix for autoregressive components."""
        X = np.ones((self.iT, self.p + 1))
        for lag in range(1, self.p + 1):
            X[:, lag] = np.roll(self.vY, lag)
        X[:self.p, 1:] = 0
        return X[self.p:]

    def tvc_quantreg(self, vTheta, u0):
        """Objective function for quantile regression."""
        vAlpha0, vAlpha1 = vTheta[: self.p + 1], vTheta[self.p + 1 :]
        kernel_weights = self.K((self.U_t - u0) / self.h) / self.h
        kernel_weights = kernel_weights[self.p:]
        residuals = self.vY[self.p:] - (self.X @ vAlpha0 + (self.X @ vAlpha1) * (self.U_t[self.p:] - u0))
        return np.sum(self.rho_tau(residuals) * kernel_weights)

    def CalcParams_u0(self, u0):
        """Optimize and calculate parameters for a specific u0 and compute qhat."""
        vTheta0 = np.zeros(2 * (self.p + 1))
        result = opt.minimize(self.tvc_quantreg, vTheta0, args=(u0,), method='SLSQP')
        vAlpha0, vAlpha1 = result.x[:self.p+1], result.x[self.p+1:]
        qhat_u0 = self.X @ vAlpha0 + (self.X @ vAlpha1) * (self.U_t[self.p:] - u0)
        return result.x, qhat_u0

    def CalcParams(self):
        """Optimize parameters for all u0 values in parallel."""
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.CalcParams_u0)(u0) for u0 in self.vU0
        )
        self.mAlpha, qhat_values = zip(*results)
        self.mAlpha = np.array(self.mAlpha)
        self.mAlpha0 = self.mAlpha[:, :self.p+1]
        self.mAlpha1 = self.mAlpha[:, self.p+1:]

        self.qhat = np.array(qhat_values)
        
        print("Parameter estimation and qhat calculation completed.")




class calc_AIC:
    def __init__(self, ctv, ctv_reference):
        """
        Initialize the CALC_AIC class with the necessary parameters.

        Parameters:
        - ctv: Object containing input data and parameters.
        - ctv_reference: Reference object for calculations.
        """
        self.ctv = ctv
        self.ctv_reference = ctv_reference
    
    def conditional_density_approximation(self, tau_vals, quantile_series_list):
        """
        Approximates the conditional density using discrete quantiles for multiple quantile series.
        """
        density_approximations = []
        for i in range(1, len(tau_vals)):
            quantile_series_lower = quantile_series_list[i - 1]
            quantile_series_upper = quantile_series_list[i]
            delta_tau = tau_vals[i] - tau_vals[i - 1]
            density_approx = [
                delta_tau / (quantile_series_upper[t] - quantile_series_lower[t])
                if quantile_series_upper[t] > quantile_series_lower[t] else 0.
                for t in range(len(quantile_series_lower))
            ]
            density_approximations.append(density_approx)
        return density_approximations

    def calc_l_S_n(self):
        """
        Calculate the l_S_n matrix for the given data.
        """
        mqhat = self.ctv.qhat
        mqhat_reference = self.ctv_reference.qhat
        l_S_n = []
        
        for j, u0 in enumerate(self.ctv.vU0):
            print(f'\rCurrently calculating {j+1} out of {len(self.ctv.vU0)}.', end='')

            S_n = np.zeros((self.ctv.X.shape[1] * 2, self.ctv.X.shape[1] * 2))
            l_f_y_xu = self.conditional_density_approximation(
                [self.ctv.fTau - 0.05, self.ctv.fTau], 
                [mqhat_reference[j, :], mqhat[j, :]]
            )[0]
            
            U_th = (self.ctv.U_t - u0) / self.ctv.h
            mUth = np.tile(U_th, (2 if self.ctv.p == 1 else 3, 1)).T
            X_array = np.array(self.ctv.X)
            X_star = np.hstack([X_array, mUth[self.ctv.p:] * X_array])
            K_values = self.ctv.K(U_th[self.ctv.p:])

            for i in range(len(self.ctv.vY) - self.ctv.p):
                X_star_row = X_star[i, :].reshape(-1, 1)
                S_n += np.outer(l_f_y_xu[i] * X_star_row, X_star_row.T) * K_values[i]

            l_S_n.append(S_n)
        return l_S_n

    def calc_p_h(self):
        """
        Calculate the p_h value.
        """
        l_S_n = self.calc_l_S_n()[self.ctv.p:]
        X_null = pd.concat([pd.DataFrame(self.ctv.X), pd.DataFrame(np.zeros_like(self.ctv.X))], axis=1)
        X_null_array = X_null.to_numpy()

        K_0 = self.ctv.K(0)
        a_n_squared = self.ctv.a_n**2
        l_gamma = []

        for i in range(len(self.ctv.vU0) - self.ctv.p):
            X_i = X_null_array[i, :].reshape(-1, 1)
            inv_l_S_n = np.linalg.inv(l_S_n[i])
            gamma_i = a_n_squared * K_0 * (X_i.T @ inv_l_S_n @ X_i).item()
            l_gamma.append(gamma_i)

        return np.sum(l_gamma)

    def calculate_AIC(self):
        """
        Calculate the Akaike Information Criterion (AIC).
        """
        l_quantsum_h = []
        qhat_u0 = np.diag(self.ctv.qhat[self.ctv.p:])

        for t in range(len(self.ctv.vY) - self.ctv.p):
            rho_val = self.ctv.rho_tau(self.ctv.vY[self.ctv.p:][t] - qhat_u0[t])
            l_quantsum_h.append(rho_val)

        quantsum_h = np.mean(l_quantsum_h)
        p_h = self.calc_p_h()
        AIC = np.log(quantsum_h) + 2 * (p_h + 1) / (self.ctv.iT - p_h - 2)
        return AIC


class calc_COV:
    def __init__(self, ctv, ctv_reference):
        """
        Initialize the CALC_AIC class with the necessary parameters.

        Parameters:
        - ctv: Object containing input data and parameters.
        - ctv_reference: Reference object for calculations.
        """
        self.ctv = ctv
        self.ctv_reference = ctv_reference
    
    def conditional_density_approximation(self, tau_vals, quantile_series_list):
        """
        Approximates the conditional density using discrete quantiles for multiple quantile series.
        """
        density_approximations = []
        for i in range(1, len(tau_vals)):
            quantile_series_lower = quantile_series_list[i - 1]
            quantile_series_upper = quantile_series_list[i]
            delta_tau = tau_vals[i] - tau_vals[i - 1]
            density_approx = [
                delta_tau / (quantile_series_upper[t] - quantile_series_lower[t])
                if quantile_series_upper[t] > quantile_series_lower[t] else 0.
                for t in range(len(quantile_series_lower))
            ]
            density_approximations.append(density_approx)
        return density_approximations

    def calc_l_Sigmahat(self):
        """
        Calculate the l_S_n matrix for the given data.
        """
        mqhat = self.ctv.qhat
        mqhat_reference = self.ctv_reference.qhat
        lSigma_hat = []
        
        for j, u0 in enumerate(self.ctv.vU0):
            print(f'\rCurrently calculating {j+1} out of {len(self.ctv.vU0)}.', end='')

            Omega_n_0 = np.zeros((self.ctv.X.shape[1], self.ctv.X.shape[1]))
            Omega_n_1 = np.zeros((self.ctv.X.shape[1], self.ctv.X.shape[1]))

            l_f_y_xu = self.conditional_density_approximation(
                [self.ctv.fTau - 0.05, self.ctv.fTau], 
                [mqhat_reference[j, :], mqhat[j, :]]
            )[0]
            
            U_th = (self.ctv.U_t - u0) / self.ctv.h
            K_values = self.ctv.K(U_th[self.ctv.p:]) / self.ctv.h

            for i in range(len(self.ctv.vY) - self.ctv.p):
                X_row = np.array(self.ctv.X)[i, :].reshape(-1, 1)
                Omega_n_1 += np.outer(l_f_y_xu[i] * X_row, X_row.T) * K_values[i]
                Omega_n_0 += np.outer(X_row, X_row.T) * K_values[i]
                
            lSigma_hat.append(np.linalg.inv(Omega_n_1) @ Omega_n_0 @ np.linalg.inv(Omega_n_1))
        l_Cov_matrix = [Sigma * self.ctv.fTau * (1 - self.ctv.fTau) * 0.6 for Sigma in lSigma_hat]
        return l_Cov_matrix


