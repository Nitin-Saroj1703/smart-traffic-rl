# Smart Traffic RL Dashboard

## Features

### 📊 Page 1: Live Simulation View
- Real-time 3x3 grid visualization of all 9 intersections
- Color-coded signal phases (Green/Yellow/Red)
- Live metrics sidebar with network-wide statistics
- Auto-updates every second

### 📈 Page 2: Performance Comparison
- Side-by-side comparison: RL Agent vs Static Timer
- Three interactive charts:
  - Wait time comparison
  - CO2 emissions comparison  
  - Throughput comparison
- Summary metric cards showing percentage improvements

### 🧠 Page 3: Agent Decision Explainer
- Select any intersection to analyze
- Visualize observation vector as bar chart
- See which action was taken and why
- Feature importance ranking
- Reward breakdown as stacked bar chart

### ⚙️ Page 4: Configuration Panel
- Adjust reward weights in real-time
- Fine-tune model with new parameters
- Export results as CSV
- View expected performance metrics

## Quick Start

1. Install dependencies:
```bash
pip install streamlit plotly pandas numpy pyyaml
```

2. Run the dashboard:
```bash
streamlit run dashboard/app.py
```

## Configuration
All application settings are stored in `dashboard/config.yaml`. You can adjust:
- Refresh rates
- Color schemes
- Intersection grid dimensions
- Default reward weightings
