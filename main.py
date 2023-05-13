# Streamlit
import streamlit as st

# Import Libraries
import numpy as np
import scipy.stats as stat
import plotly.graph_objs as go


# Black-Scholes Option Formula
def europian_option(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        V = S * stat.norm.cdf(d1) - K * np.exp((-r) * T) * stat.norm.cdf(d2)
    else:
        V = K * np.exp((-r) * T) * stat.norm.cdf(-d2) - S * stat.norm.cdf(-d1)
    return V


# Parameters
K = 242200
r = 0.0388
sigma = 0.04

# Variables
T = np.linspace(1e-10, 8, 100)
S = np.linspace(179000, 242200, 100)
T, S = np.meshgrid(T, S)

# Output
Call_Value = europian_option(S, K, T, r, sigma, 'call')

trace = go.Surface(x=T, y=S, z=Call_Value)
data = [trace]
layout = go.Layout(title='Call Option',
                   scene={'xaxis': {'title': 'Maturity'},
                          'yaxis': {'title': 'Spot Price'},
                          'zaxis': {'title': 'Option Price'}})

fig = go.Figure(data=data, layout=layout)

# Plotting with Streamlit
st.plotly_chart(fig)
