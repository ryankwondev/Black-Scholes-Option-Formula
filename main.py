# Streamlit
import streamlit as st

# Import Libraries
import numpy as np
import scipy.stats as stat
import plotly.graph_objs as go

# ========== FOR PRODUCTION USE ==========
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# ========================================

st.title('Black-Scholes Option Pricing Model')


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

# enter parameters
st.sidebar.header('Option Parameters')
K = st.sidebar.number_input('Strike Price', value=K)
r = st.sidebar.number_input('Risk Free Rate', value=r)
sigma = st.sidebar.number_input('Volatility', value=sigma)

# Variables
# T = np.linspace(1e-10, 8, 100)
# S = np.linspace(179000, 242200, 100)
# T, S = np.meshgrid(T, S)

# T/S Param
st.sidebar.header('T/S Parameters')
T_min = st.sidebar.number_input('T min', value=1e-10)
T_max = st.sidebar.number_input('T max', value=8)
S_min = st.sidebar.number_input('S min', value=179000)
S_max = st.sidebar.number_input('S max', value=242200)

# T/S Range
T = np.linspace(T_min, T_max, 100)
S = np.linspace(S_min, S_max, 100)
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
