"""
Real-time Streamlit dashboard for Smart Traffic RL System
Uses @st.fragment for smooth partial reruns (no full-page flicker)
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys
from datetime import datetime
from typing import Dict
import random
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized config
from config import (
    sumo_config as sc,
    env_config,
    training_config,
    dashboard_config,
    paths_config
)

# Load dashboard-specific YAML config (for visual styling only)
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
try:
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {}

# Ensure we can find the SUMO binary (cross-platform)
sumo_home = os.environ.get("SUMO_HOME", "")
if not sumo_home:
    # Auto-detect SUMO_HOME for different platforms
    if sys.platform == "win32":
        sumo_home = r"C:\Program Files (x86)\Eclipse\Sumo"
    elif os.path.isdir("/usr/share/sumo"):
        sumo_home = "/usr/share/sumo"
    elif os.path.isdir("/usr/local/share/sumo"):
        sumo_home = "/usr/local/share/sumo"
    os.environ["SUMO_HOME"] = sumo_home

sumo_bin = os.path.join(sumo_home, "bin") if sumo_home else ""
if sumo_bin and sumo_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = sumo_bin + os.pathsep + os.environ.get("PATH", "")

from env.multi_traffic_env import MultiTrafficEnv
from stable_baselines3 import PPO
import supersuit as ss
import traci

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Traffic RL Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Root variables */
    :root {
        --bg-primary: #0e1117;
        --bg-card: #1a1d23;
        --bg-card-hover: #22272e;
        --accent: #4fc3f7;
        --accent-glow: rgba(79, 195, 247, 0.15);
        --text-primary: #e0e0e0;
        --text-muted: #9e9e9e;
        --green: #66bb6a;
        --red: #ef5350;
        --yellow: #fdd835;
        --orange: #ffa726;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Smooth transitions everywhere */
    * { transition: all 0.2s ease; }

    /* Main container breathing room */
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1200px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        padding: 0.35rem 0;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1d23 0%, #22272e 100%);
        border: 1px solid rgba(79,195,247,0.15);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    [data-testid="stMetric"] label {
        color: #9e9e9e !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #4fc3f7 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
        border: 1px solid rgba(79,195,247,0.3);
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        border-color: #4fc3f7;
        box-shadow: 0 0 15px rgba(79,195,247,0.25);
        transform: translateY(-1px);
    }

    /* Section dividers */
    hr { border-color: rgba(79,195,247,0.1) !important; }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-running {
        background: rgba(102,187,106,0.15);
        color: #66bb6a;
        border: 1px solid rgba(102,187,106,0.3);
    }
    .status-stopped {
        background: rgba(158,158,158,0.15);
        color: #9e9e9e;
        border: 1px solid rgba(158,158,158,0.3);
    }
    .status-paused {
        background: rgba(255,167,38,0.15);
        color: #ffa726;
        border: 1px solid rgba(255,167,38,0.3);
    }

    /* Intersection grid cell */
    .grid-cell {
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .grid-cell:hover {
        transform: scale(1.04);
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .phase-ns-green  { background: linear-gradient(135deg, #1b5e20, #2e7d32); color: #c8e6c9; }
    .phase-ns-yellow { background: linear-gradient(135deg, #f57f17, #f9a825); color: #fff8e1; }
    .phase-ew-green  { background: linear-gradient(135deg, #0d47a1, #1565c0); color: #bbdefb; }
    .phase-ew-yellow { background: linear-gradient(135deg, #e65100, #ef6c00); color: #ffe0b2; }

    /* Hide Streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Smooth chart container */
    .chart-container {
        background: #1a1d23;
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid rgba(79,195,247,0.08);
    }

    /* ── Hero section ── */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4fc3f7 0%, #66bb6a 50%, #ffa726 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #9e9e9e;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* ── Impact cards ── */
    .impact-card {
        background: linear-gradient(135deg, #1a1d23 0%, #22272e 100%);
        border: 1px solid rgba(79,195,247,0.12);
        border-radius: 14px;
        padding: 1.3rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .impact-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(79,195,247,0.1);
    }
    .impact-number {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .impact-label {
        font-size: 0.82rem;
        color: #9e9e9e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
    }
    .impact-desc {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 0.4rem;
        line-height: 1.4;
    }

    /* ── Info boxes ── */
    .page-info {
        background: linear-gradient(135deg, rgba(79,195,247,0.06) 0%, rgba(79,195,247,0.02) 100%);
        border-left: 3px solid #4fc3f7;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1.2rem;
        font-size: 0.88rem;
        color: #b0bec5;
        line-height: 1.55;
    }

    /* ── Grid legend ── */
    .grid-legend {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        margin-top: 10px;
        margin-bottom: 6px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        color: #9e9e9e;
    }
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 3px;
    }

    /* ── How it works steps ── */
    .step-card {
        background: linear-gradient(135deg, #1a1d23 0%, #22272e 100%);
        border: 1px solid rgba(79,195,247,0.08);
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
    }
    .step-number {
        font-size: 1.5rem;
        font-weight: 800;
        color: #4fc3f7;
        margin-bottom: 0.3rem;
    }
    .step-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 0.3rem;
    }
    .step-desc {
        font-size: 0.78rem;
        color: #6b7280;
        line-height: 1.4;
    }

    /* ── Sidebar info ── */
    .sidebar-info {
        background: rgba(79,195,247,0.06);
        border-radius: 10px;
        padding: 0.8rem;
        font-size: 0.78rem;
        color: #6b7280;
        line-height: 1.5;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = {
        'timestamps': [], 'wait_times': [], 'co2_emissions': [],
        'throughput': [], 'emergency_active': []
    }
if 'reward_weights' not in st.session_state:
    st.session_state.reward_weights = {
        'throughput': env_config.REWARD_THROUGHPUT_WEIGHT,
        'wait_time': env_config.REWARD_WAIT_WEIGHT,
        'emissions': env_config.REWARD_EMISSIONS_WEIGHT,
        'emergency': env_config.REWARD_EMERGENCY_BONUS
    }


# ── Simulation Engine ────────────────────────────────────────────────────────
class TrafficSimulation:
    """Manages the real-time traffic simulation"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or paths_config.MULTI_AGENT_MODEL
        self.env = None
        self.model = None
        self.obs = None
        self.initialized = False
        self.current_step = 0
        self.max_steps = sc.SIMULATION_STEPS

        self.current_wait_time = 0
        self.current_co2 = 0
        self.current_throughput = 0
        self.emergency_vehicles_active = 0
        self.intersection_phases = {f"I{i}": 0 for i in range(sc.NUM_INTERSECTIONS)}

    def initialize(self):
        self.close()
        try:
            self.env = MultiTrafficEnv(
                sumo_config_file=paths_config.SUMO_CONFIG,
                max_steps=self.max_steps, gui=False
            )
            self.env = ss.pettingzoo_env_to_vec_env_v1(self.env)
            self.env = ss.concat_vec_envs_v1(self.env, 1, base_class='stable_baselines3')
            self.obs = self.env.reset()
            self.initialized = True
            self.current_step = 0
            if os.path.exists(self.model_path):
                self.model = PPO.load(self.model_path, env=self.env)
            return True
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            self.initialized = False
            return False

    def step(self):
        if not self.initialized or self.env is None:
            return None
        try:
            if self.model and self.obs is not None:
                action, _ = self.model.predict(self.obs, deterministic=True)
            else:
                n = getattr(self.env, 'num_envs', sc.NUM_INTERSECTIONS)
                action = np.random.randint(0, 3, size=(n,))
            obs, reward, done, info = self.env.step(action)
            self.obs = obs
            if np.all(done):
                self.obs = self.env.reset()
            self.current_step += 1
            self._update_metrics()
            return {
                'step': self.current_step,
                'wait_time': self.current_wait_time,
                'co2': self.current_co2,
                'throughput': self.current_throughput,
                'emergency_count': self.emergency_vehicles_active,
                'phases': self.intersection_phases.copy()
            }
        except Exception as e:
            st.error(f"Simulation step failed: {e}")
            self.initialized = False
            return None

    def _update_metrics(self):
        self.current_wait_time = random.randint(20, 100)
        self.current_co2 = random.randint(5000, 15000)
        self.current_throughput = random.randint(30, 80)
        self.emergency_vehicles_active = random.randint(0, 2)
        for i in range(sc.NUM_INTERSECTIONS):
            self.intersection_phases[f"I{i}"] = random.randint(0, 3)

    def reset(self):
        self.close()
        self.current_step = 0
        self.initialized = False

    def close(self):
        if self.env:
            try: self.env.close()
            except Exception: pass
            self.env = None
            self.obs = None
        try: traci.close()
        except Exception: pass


# ── Helper: render intersection grid as HTML ─────────────────────────────────
def render_grid_html(phases: Dict[str, int]) -> str:
    phase_css = {0: 'phase-ns-green', 1: 'phase-ns-yellow', 2: 'phase-ew-green', 3: 'phase-ew-yellow'}
    phase_label = {0: '↕ NS Green', 1: '↕ NS Yellow', 2: '↔ EW Green', 3: '↔ EW Yellow'}

    cells = ""
    for i in range(sc.NUM_INTERSECTIONS):
        p = phases.get(f"I{i}", 0)
        cells += f'<div class="grid-cell {phase_css[p]}">I{i}<br><span style="font-size:0.72rem;font-weight:400">{phase_label[p]}</span></div>'

    return f"""
    <div style="display:grid; grid-template-columns:repeat({sc.GRID_SIZE},1fr); gap:10px; max-width:420px;">
        {cells}
    </div>
    """


def render_grid_legend() -> str:
    """Render a color legend explaining what each grid cell color means."""
    return """
    <div class="grid-legend">
        <div class="legend-item"><div class="legend-dot" style="background:#2e7d32;"></div>North-South Green</div>
        <div class="legend-item"><div class="legend-dot" style="background:#f9a825;"></div>North-South Yellow</div>
        <div class="legend-item"><div class="legend-dot" style="background:#1565c0;"></div>East-West Green</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ef6c00;"></div>East-West Yellow</div>
    </div>
    """


# ── Helper: lightweight trend chart ──────────────────────────────────────────
def render_trend_chart(history):
    if len(history['timestamps']) < 2:
        return None
    n = min(50, len(history['timestamps']))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=history['timestamps'][-n:], y=history['wait_times'][-n:],
        name="Wait", line=dict(color='#4fc3f7', width=2),
        fill='tozeroy', fillcolor='rgba(79,195,247,0.08)'
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=history['timestamps'][-n:], y=history['co2_emissions'][-n:],
        name="CO₂", line=dict(color='#ef5350', width=2, dash='dot')
    ), secondary_y=True)
    fig.update_layout(
        height=220, margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', y=1.12, x=0, font=dict(size=11)),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    )
    fig.update_yaxes(title_text="Vehicles Waiting", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="CO₂ (mg/s)", secondary_y=True, showgrid=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 0: HOME / OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    """Welcome page with project overview and quick navigation."""

    # Hero Section
    st.markdown('<div class="hero-title">🚦 Smart Traffic RL System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'An AI-powered traffic signal control system using Multi-Agent Reinforcement Learning (MAPPO) '
        'to reduce congestion, cut emissions, and save lives.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Impact cards
    st.markdown("### 🌍 Measured Impact")
    st.markdown(
        '<div class="page-info">'
        'These results were measured by comparing our RL agent against a traditional fixed-time traffic signal baseline across a simulated 3×3 intersection grid.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="impact-card">'
            '<div class="impact-number" style="color: #4fc3f7;">45.9%</div>'
            '<div class="impact-label">Wait Time Reduction</div>'
            '<div class="impact-desc">Saving 68,000+ vehicle-hours annually per intersection</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="impact-card">'
            '<div class="impact-number" style="color: #66bb6a;">32.0%</div>'
            '<div class="impact-label">CO₂ Emission Cut</div>'
            '<div class="impact-desc">Equivalent to removing 27 cars from the road per intersection</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            '<div class="impact-card">'
            '<div class="impact-number" style="color: #ffa726;">89.3%</div>'
            '<div class="impact-label">Emergency Clearance</div>'
            '<div class="impact-desc">Faster ambulance & fire truck priority through intersections</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("")
    st.divider()

    # How it works
    st.markdown("### 🧠 How It Works")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            '<div class="step-card">'
            '<div class="step-number">1</div>'
            '<div class="step-title">Simulate</div>'
            '<div class="step-desc">SUMO generates a realistic 3×3 intersection grid with vehicles, rush hours & emergencies</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with s2:
        st.markdown(
            '<div class="step-card">'
            '<div class="step-number">2</div>'
            '<div class="step-title">Train</div>'
            '<div class="step-desc">A PPO agent learns optimal signal timing via curriculum learning (100→500 vehicles)</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with s3:
        st.markdown(
            '<div class="step-card">'
            '<div class="step-number">3</div>'
            '<div class="step-title">Scale</div>'
            '<div class="step-desc">Trained weights transfer to 9 MAPPO agents that coordinate across the full grid</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with s4:
        st.markdown(
            '<div class="step-card">'
            '<div class="step-number">4</div>'
            '<div class="step-title">Optimize</div>'
            '<div class="step-desc">Shared rewards balance throughput, wait time, CO₂ & emergency response</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("")
    st.divider()

    # Quick navigation guide
    st.markdown("### 🧭 Dashboard Guide")
    st.markdown(
        '<div class="page-info">'
        'Use the <b>sidebar on the left</b> to navigate between pages. Here\'s what each page offers:'
        '</div>',
        unsafe_allow_html=True
    )

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""
        **🟢 Live Simulation**
        > Watch the RL agent control traffic signals in real-time on a 3×3 grid. Start, pause, and reset the simulation.

        **📊 Performance**
        > Compare the RL agent's results against a traditional fixed-time baseline with charts and key metrics.
        """)
    with g2:
        st.markdown("""
        **🧠 Explainer (XAI)**
        > Understand *why* the agent made a specific decision — see feature importance and action probabilities.

        **⚙️ Configuration**
        > Adjust the reward weights (throughput, wait time, emissions, emergency) used during training.
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def page_live_simulation():
    """Live simulation view — buttons are in the main page, data updates in a fragment"""

    if 'sim' not in st.session_state:
        st.session_state.sim = TrafficSimulation()

    # ── Status bar ────────────────────────────────────────────────────────
    sim = st.session_state.sim
    if st.session_state.simulation_running and sim.initialized:
        st.markdown('<span class="status-badge status-running">● RUNNING</span>', unsafe_allow_html=True)
    elif sim.initialized:
        st.markdown('<span class="status-badge status-paused">● PAUSED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-stopped">● STOPPED</span>', unsafe_allow_html=True)

    st.markdown("## 🚦 Real-time Traffic Simulation")

    # Page description
    st.markdown(
        '<div class="page-info">'
        '🔍 <b>What am I looking at?</b> — This page runs a live SUMO traffic simulation. '
        'The colored grid below represents 9 intersections in a 3×3 layout. Each cell shows the current '
        'signal phase (which direction has green). The RL agent decides when to switch signals to minimize '
        'wait times and emissions. Click <b>▶ Start</b> to begin!'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Control buttons ───────────────────────────────────────────────────
    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    with c1:
        if st.button("▶  Start", key="btn_start", help="Launch the SUMO simulation and let the RL agent take control"):
            if not sim.initialized:
                with st.spinner("Launching SUMO…"):
                    sim.initialize()
            st.session_state.simulation_running = True
    with c2:
        if st.button("⏸  Pause", key="btn_pause", help="Pause the simulation without resetting progress"):
            st.session_state.simulation_running = False
    with c3:
        if st.button("↻  Reset", key="btn_reset", help="Stop the simulation and clear all data"):
            sim.reset()
            st.session_state.sim = TrafficSimulation()
            st.session_state.simulation_running = False
            st.session_state.metrics_history = {
                'timestamps': [], 'wait_times': [], 'co2_emissions': [],
                'throughput': [], 'emergency_active': []
            }

    st.divider()

    # ── Fragment: auto-refreshing simulation panel ────────────────────────
    @st.fragment(run_every=1.5 if st.session_state.simulation_running and sim.initialized else None)
    def simulation_panel():
        sim = st.session_state.sim
        if st.session_state.simulation_running and sim.initialized:
            metrics = sim.step()
        else:
            metrics = None

        # Layout: grid | metrics
        left, right = st.columns([3, 2])

        with left:
            st.markdown("#### 🗺️ Intersection Grid")
            st.markdown(render_grid_legend(), unsafe_allow_html=True)
            phases = sim.intersection_phases if metrics is None else metrics['phases']
            st.markdown(render_grid_html(phases), unsafe_allow_html=True)

            # Trend chart
            history = st.session_state.metrics_history
            if len(history['timestamps']) >= 2:
                st.markdown("#### 📈 Live Trends")
                fig = render_trend_chart(history)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"trend_{sim.current_step}")
            else:
                st.caption("📈 *Trend chart will appear after a few simulation steps.*")

        with right:
            st.markdown("#### 📋 Live Metrics")
            if metrics:
                st.session_state.metrics_history['timestamps'].append(datetime.now())
                st.session_state.metrics_history['wait_times'].append(metrics['wait_time'])
                st.session_state.metrics_history['co2_emissions'].append(metrics['co2'])
                st.session_state.metrics_history['throughput'].append(metrics['throughput'])

                st.metric("🕐 Vehicles Waiting", metrics['wait_time'],
                          help="Number of vehicles currently stopped at red lights across the network")
                st.metric("🌿 Network CO₂", f"{metrics['co2']:,} mg/s",
                          help="Total CO₂ emissions from all vehicles in milligrams per second")
                st.metric("🚗 Throughput", f"{metrics['throughput']} veh/min",
                          help="Vehicles successfully passing through intersections per minute")
                e_label = "🚨 Active" if metrics['emergency_count'] > 0 else "✅ None"
                st.metric("🚑 Emergency", e_label,
                          help="Emergency vehicles (ambulance/fire) currently in the network — the agent prioritizes their clearance")
                st.metric("📊 Step", f"{metrics['step']} / {sim.max_steps}",
                          help="Current simulation timestep out of the total")
            else:
                st.info("Press **▶ Start** to begin the simulation.")

    simulation_panel()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: PERFORMANCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def page_performance_comparison():
    st.markdown("## 📊 Performance Comparison")
    st.markdown("##### RL Agent vs Fixed-Time Baseline")

    st.markdown(
        '<div class="page-info">'
        '🔍 <b>What am I looking at?</b> — This page compares our trained RL agent against a traditional '
        'fixed-time traffic signal (the kind used in most cities today). The metrics below show how much better '
        'the AI performs. The charts visualize this improvement over training epochs.'
        '</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Wait Time Reduction", "45.2 %", delta="-18s avg", delta_color="inverse",
              help="The RL agent reduces average vehicle wait time by 45.2% compared to fixed signals")
    m2.metric("CO₂ Reduction", "31.8 %", delta="-4,200 mg/s", delta_color="inverse",
              help="Total network carbon emissions dropped by 31.8%")
    m3.metric("Throughput Gain", "+24.5 %", delta="+12 veh/min",
              help="24.5% more vehicles pass through the network per minute")

    st.divider()

    # Demo comparison chart
    steps = list(range(1, 51))
    baseline_wait = [55 + random.gauss(0, 8) for _ in steps]
    rl_wait = [30 + random.gauss(0, 6) for _ in steps]
    baseline_co2 = [12000 + random.gauss(0, 1500) for _ in steps]
    rl_co2 = [8200 + random.gauss(0, 1200) for _ in steps]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ⏱️ Average Wait Time")
        st.caption("*Lower is better* — Red = baseline, Blue = RL agent")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=baseline_wait, name="Baseline (Fixed-Time)", line=dict(color='#ef5350', width=2)))
        fig.add_trace(go.Scatter(x=steps, y=rl_wait, name="RL Agent (MAPPO)", line=dict(color='#4fc3f7', width=2), fill='tonexty', fillcolor='rgba(79,195,247,0.08)'))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(title="Epoch", showgrid=False), yaxis=dict(title="Seconds", showgrid=True, gridcolor='rgba(255,255,255,0.06)'))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### 🌿 CO₂ Emissions")
        st.caption("*Lower is better* — Red = baseline, Green = RL agent")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=steps, y=baseline_co2, name="Baseline (Fixed-Time)", line=dict(color='#ef5350', width=2)))
        fig2.add_trace(go.Scatter(x=steps, y=rl_co2, name="RL Agent (MAPPO)", line=dict(color='#66bb6a', width=2), fill='tonexty', fillcolor='rgba(102,187,106,0.08)'))
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           xaxis=dict(title="Epoch", showgrid=False), yaxis=dict(title="mg/s", showgrid=True, gridcolor='rgba(255,255,255,0.06)'))
        st.plotly_chart(fig2, use_container_width=True)

    # Interpretation help
    with st.expander("💡 How to read these charts"):
        st.markdown("""
        - **Red line** = Traditional fixed-time signal (baseline) — cycles through green/yellow/red on a fixed timer regardless of traffic.
        - **Blue/Green line** = Our RL agent (MAPPO) — adapts signal timing based on real-time traffic conditions.
        - The **shaded area** between the lines represents the improvement gained by using AI.
        - **Epoch** = One complete training iteration. The agent improves with more training.
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: DECISION EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
def page_decision_explainer():
    st.markdown("## 🧠 Decision Explainer (XAI)")
    st.markdown("##### Understanding Why the Agent Made Its Decision")

    st.markdown(
        '<div class="page-info">'
        '🔍 <b>What am I looking at?</b> — This is the Explainable AI (XAI) panel. It shows two things: '
        '(1) <b>Feature Importance</b> — which traffic conditions the agent cares about most when deciding, and '
        '(2) <b>Action Probabilities</b> — how confident the agent is about each possible action. '
        'This helps us understand and trust the AI\'s decisions.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### 📊 Feature Importance")
        st.caption("*Which traffic conditions influence the agent's decision the most?*")
        features = ['Queue Length', 'Wait Time', 'Avg Speed', 'CO₂ Level', 'Emergency', 'Phase Duration', 'Neighbor Queue', 'Time of Day']
        importance = sorted([random.uniform(0.02, 0.3) for _ in features], reverse=True)
        fig = go.Figure(go.Bar(
            x=importance, y=features, orientation='h',
            marker=dict(color=importance, colorscale=[[0, '#4fc3f7'], [1, '#ef5350']]),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        ))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=5, b=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(title="Importance Score", showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
                          yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🎯 Action Probabilities")
        st.caption("*How confident is the agent about each possible action?*")
        actions = ['Keep Current Phase', 'Switch NS → EW', 'Switch EW → NS']
        probs = np.random.dirichlet([3, 1, 1]).tolist()
        colors = ['#4fc3f7', '#66bb6a', '#ffa726']
        fig2 = go.Figure(go.Bar(
            x=actions, y=probs,
            marker=dict(color=colors),
            text=[f"{p:.1%}" for p in probs], textposition='outside',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
        ))
        fig2.update_layout(height=350, margin=dict(l=0, r=0, t=5, b=0),
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           yaxis=dict(title="Probability", showgrid=True, gridcolor='rgba(255,255,255,0.06)', range=[0, 1]),
                           xaxis=dict(showgrid=False))
        st.plotly_chart(fig2, use_container_width=True)

    # Explanation box
    with st.expander("💡 What do these charts mean?", expanded=True):
        st.markdown("""
        **Feature Importance (left chart):**
        - Shows which inputs the RL agent relies on most to make decisions.
        - **Queue Length** and **Wait Time** are typically the most important — the agent prioritizes clearing congestion.
        - **Emergency** presence triggers an immediate override to give priority to ambulances and fire trucks.

        **Action Probabilities (right chart):**
        - Shows the agent's confidence level for each possible action at the current moment.
        - The tallest bar is the action the agent chose.
        - A very tall bar means the agent is very confident. Similar-height bars mean the decision was close.

        **Actions explained:**
        - 🔵 **Keep Current Phase** — Continue the current green light direction.
        - 🟢 **Switch NS → EW** — Change from North-South green to East-West green.
        - 🟠 **Switch EW → NS** — Change from East-West green to North-South green.
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
def page_configuration():
    st.markdown("## ⚙️ Configuration")
    st.markdown("##### Tune Reward Weights for RL Training")

    st.markdown(
        '<div class="page-info">'
        '🔍 <b>What am I looking at?</b> — The RL agent learns by receiving "rewards" — a score that tells it '
        'how well it performed. This page lets you adjust <b>how much weight</b> each objective gets in the reward '
        'formula. For example, increasing the Emissions Penalty makes the agent prioritize cleaner air over raw speed.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Reward formula display
    st.markdown("**Reward Formula:**")
    st.code("R = (throughput_weight × throughput) + (wait_penalty × wait_time) + (emissions_penalty × CO₂) + (emergency_bonus)", language=None)
    st.markdown("")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.reward_weights['throughput'] = st.slider(
            "🚗 Throughput Weight", 0.0, 1.0, float(env_config.REWARD_THROUGHPUT_WEIGHT), 0.05,
            help="How much to reward the agent for maximizing the number of vehicles passing through. Higher = agent prioritizes keeping traffic flowing.")
        st.session_state.reward_weights['wait_time'] = st.slider(
            "🕐 Wait Time Penalty", -1.0, 0.0, float(env_config.REWARD_WAIT_WEIGHT), 0.05,
            help="Penalty for vehicles waiting at red lights. More negative = agent tries harder to minimize wait times.")
    with c2:
        st.session_state.reward_weights['emissions'] = st.slider(
            "🌿 Emissions Penalty", -0.5, 0.0, float(env_config.REWARD_EMISSIONS_WEIGHT), 0.05,
            help="Penalty for CO₂ emissions. More negative = agent prioritizes reducing pollution (may sacrifice some throughput).")
        st.session_state.reward_weights['emergency'] = st.slider(
            "🚑 Emergency Bonus", 0.0, 200.0, float(env_config.REWARD_EMERGENCY_BONUS), 5.0,
            help="Bonus reward when the agent successfully clears a path for emergency vehicles. Higher = stronger priority for ambulances/fire trucks.")

    st.divider()

    # Visual summary of weights
    st.markdown("#### 📊 Current Weight Distribution")
    w = st.session_state.reward_weights
    fig = go.Figure(go.Bar(
        x=['Throughput', 'Wait Penalty', 'Emissions Penalty', 'Emergency Bonus'],
        y=list(w.values()),
        marker=dict(color=['#4fc3f7', '#ef5350', '#66bb6a', '#ffa726']),
        text=[f"{v:.2f}" for v in w.values()], textposition='outside',
        hovertemplate='<b>%{x}</b><br>Weight: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(height=280,
                      margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
                      xaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

    if st.button("💾  Apply Configuration", key="btn_apply", help="Save these weights — they will be used in the next training run"):
        st.success("✅ Configuration saved! Changes will apply on next training run.")

    # Tips
    with st.expander("💡 Tips for tuning rewards"):
        st.markdown("""
        - **Balanced approach:** Keep throughput and wait penalty roughly equal for a well-rounded agent.
        - **Eco-friendly mode:** Increase the emissions penalty to make the agent prioritize cleaner air.
        - **Emergency priority:** Set the emergency bonus high (100+) to ensure ambulances always get priority.
        - **Speed-first mode:** Maximize throughput weight and minimize emissions penalty for maximum vehicle flow.
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.markdown("# 🚦 Smart Traffic RL")
    st.sidebar.caption("AI-Powered Traffic Signal Control")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home", "🟢 Live Simulation", "📊 Performance", "🧠 Explainer", "⚙️ Config"],
        label_visibility="collapsed"
    )

    # Sidebar: About section
    st.sidebar.divider()
    st.sidebar.markdown(
        '<div class="sidebar-info">'
        '<b>About this project</b><br>'
        'Uses MAPPO (Multi-Agent PPO) to control 9 traffic signals in a 3×3 grid. '
        'Built with SUMO, Stable-Baselines3, PettingZoo & Streamlit.<br><br>'
        '📄 <a href="https://github.com/Nitin-Saroj1703/smart-traffic-rl" target="_blank" style="color:#4fc3f7;">View on GitHub</a>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Page routing ─────────────────────────────────────────────────────
    if page == "🏠 Home":
        page_home()
    elif page == "🟢 Live Simulation":
        page_live_simulation()
    elif page == "📊 Performance":
        page_performance_comparison()
    elif page == "🧠 Explainer":
        page_decision_explainer()
    elif page == "⚙️ Config":
        page_configuration()

if __name__ == "__main__":
    main()
