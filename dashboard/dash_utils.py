"""
Utility functions for dashboard
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import streamlit as st

def calculate_metrics(history: Dict) -> Dict:
    """Calculate summary metrics from history"""
    if not history['wait_times']:
        return {}
    
    return {
        'avg_wait': np.mean(history['wait_times'][-50:]),
        'avg_co2': np.mean(history['co2_emissions'][-50:]),
        'avg_throughput': np.mean(history['throughput'][-50:]),
        'peak_wait': np.max(history['wait_times']),
        'emergency_freq': sum(history['emergency_active']) / len(history['emergency_active']) if history['emergency_active'] else 0
    }

def create_heatmap(data: np.ndarray, title: str) -> go.Figure:
    """Create a heatmap visualization"""
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale='RdYlGn_r',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="Column",
        yaxis_title="Row"
    )
    
    return fig

@st.cache_data
def load_historical_data(days: int = 7) -> pd.DataFrame:
    """Load historical performance data"""
    # Generate sample data for demonstration
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days*24*60, freq='1min')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'wait_time': 50 + 30 * np.sin(np.arange(len(dates)) * 0.01) + np.random.randn(len(dates)) * 10,
        'co2_emissions': 8000 + 2000 * np.sin(np.arange(len(dates)) * 0.01) + np.random.randn(len(dates)) * 500,
        'throughput': 60 + 20 * np.cos(np.arange(len(dates)) * 0.005) + np.random.randn(len(dates)) * 15
    })
    
    return df
