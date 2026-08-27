# 🚦 Smart Traffic RL — Adaptive Signal Control

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![SUMO](https://img.shields.io/badge/SUMO-1.19+-4CAF50?style=for-the-badge&logo=eclipse&logoColor=white)](https://www.eclipse.org/sumo/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

> Multi-agent reinforcement learning system for adaptive traffic signal control, trained on a 3×3 SUMO intersection grid using **MAPPO** (Multi-Agent Proximal Policy Optimization).

🔗 **[Live Demo →](https://nitin-saroj1703.github.io/smart-traffic-rl/)**

---

## 🌍 Why This Matters

Urban traffic congestion costs the global economy **\$1.5 trillion annually** and contributes to **23% of CO₂ emissions** worldwide. This project demonstrates that AI-powered traffic control can make a real difference:

| Metric | Fixed-Time Baseline | RL Agent | Improvement |
|---|---|---|---|
| **Avg Wait Time** | 17.0 vehicles | 9.2 vehicles | **↓ 45.9%** |
| **CO₂ Emissions** | 12,500 mg/s | 8,500 mg/s | **↓ 32.0%** |
| **Throughput** | 55.9 veh/min | 68.5 veh/min | **↑ 22.5%** |
| **Emergency Clearance** | 45.0% | 85.2% | **↑ 89.3%** |

> Equivalent to saving **68,000+ vehicle-hours/year** and removing **27 cars** worth of emissions per intersection.

---

## 🏗️ Architecture

```
                 ┌────────────────────────────┐
                 │      SUMO Simulation       │
                 │    3×3 Intersection Grid    │
                 └──────────┬─────────────────┘
                            │ TraCI API
                 ┌──────────▼─────────────────┐
                 │   Gymnasium / PettingZoo    │
                 │   Multi-Agent Environment   │
                 └──────────┬─────────────────┘
                            │ Observations (18-dim)
              ┌─────────────▼──────────────────┐
              │     MAPPO (9 Agents, PPO)       │
              │  Shared global + local rewards  │
              │    Policy: [256, 256] MLP        │
              └─────────────┬──────────────────┘
                            │ Actions
                   ┌────────▼────────┐
                   │  Signal Phases  │
                   │  Keep / Switch  │
                   │  / Emergency    │
                   └─────────────────┘
```

### Training Pipeline

1. **SUMO simulation** generates realistic traffic on a 3×3 grid — rush hours, random flows, emergency vehicles
2. A **PPO agent** is curriculum-trained on the center intersection (`n11`), progressing from 100 → 300 → 500 vehicles
3. Trained weights transfer to **9 MAPPO agents**, one per intersection, with shared global rewards (40% global + 60% local)
4. **Reward function** balances throughput, wait times, emissions, and emergency priority

### Reward Function

```
R = 0.4 × throughput − 0.4 × wait_penalty − 0.2 × CO₂ + emergency_bonus
```

---

## 📁 Project Structure

```
smart-traffic-rl/
├── main.py                    # Menu-driven entry point (train, eval, test)
├── config.py                  # Central hyperparameters & path configuration
├── requirements.txt           # Python dependencies
│
├── env/                       # Custom Gymnasium & PettingZoo environments
│   ├── traffic_env.py         #   Single-agent traffic signal environment
│   ├── multi_traffic_env.py   #   Multi-agent (9 intersections) environment
│   └── traffic_signal_env.py  #   Base signal environment
│
├── training/                  # Training pipelines
│   ├── train_single.py        #   PPO curriculum training (3 stages)
│   ├── train_multi.py         #   MAPPO multi-agent training
│   └── evaluate.py            #   Evaluation against baselines
│
├── agents/                    # Trained model checkpoints (.zip)
│   ├── ppo_stage1.zip         #   Stage 1: 100 vehicles
│   ├── ppo_stage2.zip         #   Stage 2: 300 vehicles
│   └── ppo_stage3.zip         #   Stage 3: 500 vehicles
│
├── simulation/                # SUMO network & route definitions
│   ├── generate_network.py    #   Auto-generates the 3×3 grid
│   └── networks/              #   Generated .net.xml, .rou.xml, .sumocfg
│
├── tests/                     # Test suites
│   ├── test_sumo.py           #   SUMO connectivity tests
│   ├── test_env.py            #   Environment unit tests
│   └── test_adversarial.py    #   Robustness & adversarial tests
│
├── utils/                     # Helpers & SUMO utilities
├── models/                    # Additional model artifacts
├── docs/                      # Static frontend (GitHub Pages)
│   ├── index.html             #   Dashboard UI
│   ├── style.css              #   Styles
│   └── app.js                 #   Simulation, charts & interactivity
│
├── run_training.py            # Quick-start training script
├── run_multi_agent.py         # Quick-start multi-agent training
├── run_env_test.py            # Quick-start environment test
└── Smart_Traffic_RL_Manual.pdf
```

---

## 📦 Installation

### Prerequisites

- **Python 3.8+**
- **SUMO 1.19+** ([download](https://sumo.dlr.de/docs/Downloads.php))
- **pip** package manager

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Nitin-Saroj1703/smart-traffic-rl.git
cd smart-traffic-rl

# 2. Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Install SUMO

<details>
<summary><b>Windows</b></summary>

Download from [sumo.dlr.de](https://sumo.dlr.de/docs/Downloads.php) and set the environment variable:
```powershell
[System.Environment]::SetEnvironmentVariable("SUMO_HOME", "C:\Program Files (x86)\Eclipse\Sumo", "User")
```
</details>

<details>
<summary><b>Ubuntu / Debian</b></summary>

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update && sudo apt-get install sumo sumo-tools
```
</details>

<details>
<summary><b>macOS</b></summary>

```bash
brew install sumo
```
</details>

---

## 🎮 Usage

### Interactive Menu

```bash
python main.py
```

```
┌─────────────────────────────────────────────────────────────┐
│                       MAIN MENU                             │
├─────────────────────────────────────────────────────────────┤
│  1. Train Single Agent (PPO with Curriculum Learning)       │
│  2. Train Multi-Agent (MAPPO for 9 Intersections)           │
│  3. Run Evaluation Only                                     │
│  4. Run Adversarial Tests                                   │
│  5. Generate SUMO Network                                   │
│  6. Run All Tests                                           │
└─────────────────────────────────────────────────────────────┘
```

### Command-Line Flags

```bash
python main.py --mode train    # Train single PPO agent (curriculum)
python main.py --mode multi    # Train 9 MAPPO agents on 3×3 grid
python main.py --mode eval     # Evaluate against fixed-time baseline
python main.py --mode test     # Run all test suites
```

### Quick-Start Scripts

```bash
python run_training.py         # Start single-agent training
python run_multi_agent.py      # Start multi-agent training
python run_env_test.py         # Verify environment setup
```

---

## 🧠 How It Works

### Observation Space (18 dimensions per agent)

Each agent observes its local intersection state:

| Feature | Description |
|---|---|
| Queue lengths (4) | Vehicles queued per approach |
| Average speeds (4) | Mean speed per approach lane |
| CO₂ levels (4) | Emissions per approach |
| Current phase (1) | Active signal phase |
| Phase duration (1) | Time in current phase |
| Emergency flags (4) | Emergency vehicle presence per approach |

### Action Space (3 actions)

| Action | Effect |
|---|---|
| `0` — Keep phase | Maintain current signal |
| `1` — Switch phase | Toggle NS ↔ EW green |
| `2` — Emergency | Override for emergency clearance |

### Multi-Agent Coordination

- **Adjacency matrix** defines which intersections share reward signals
- **Reward**: 60% local + 40% global → agents cooperate to prevent queue propagation
- **Weight transfer**: Pre-trained single-agent weights bootstrap multi-agent training

---

## 🧪 Adversarial Robustness

| Scenario | Result | Verdict |
|---|---|---|
| **Lane blockage** (accident) | 1.23× queue increase | ✅ Excellent |
| **Sensor failure** (2 lanes) | 18.5% performance drop | ✅ Graceful degradation |
| **Emergency vehicles** | 85.2% clearance, 28.3s avg | ✅ Strong |

---

## 🛠️ Tech Stack

| Category | Library | Version |
|---|---|---|
| RL Framework | Stable-Baselines3 | ≥ 2.3.0 |
| Multi-Agent | PettingZoo | 1.24.1 |
| Multi-Agent Wrapper | SuperSuit | 3.9.0 |
| Simulation | SUMO + TraCI | ≥ 1.19 |
| Deep Learning | PyTorch | ≥ 2.0 |
| Visualization | Plotly | ≥ 5.17 |
| Data | NumPy / Pandas | ≥ 1.24 / ≥ 2.0 |

---

## 🚀 Real-World Deployment Roadmap

This project currently runs on **SUMO simulation**. Here's how it can be extended to control real-world traffic signals.

### Simulation vs Real-World

| Aspect | Current (SUMO) | Real-World |
|---|---|---|
| **Traffic data** | Simulated vehicles | Camera / sensor feeds (CCTV, LiDAR, inductive loops) |
| **Signal control** | SUMO's TraCI API | Hardware controller (SCATS, SCOOT, or custom PLC) |
| **Observations** | Perfect state from simulator | Noisy, incomplete sensor data |
| **Latency** | Instant | Network delays, processing time |
| **Safety** | No consequences | Human lives at stake |

### What Would Change

#### 1. Replace SUMO with Real Sensor Input
Instead of `traci.lane.getLastStepVehicleNumber()`, real-world systems use:
- **CCTV + Computer Vision** (YOLO / OpenCV) → vehicle counting, queue detection
- **Inductive loop detectors** → embedded in roads, count passing vehicles
- **LiDAR / Radar** → measure speed, vehicle density
- **Google Maps / HERE Traffic API** → live traffic flow data

#### 2. Replace TraCI Actions with Hardware Control
Instead of `traci.trafficlight.setPhase()`, interface with:
- A **traffic signal controller** (hardware box at the intersection)
- Communication protocols like **NTCIP** (US) or **UTMC** (UK)
- An **edge computing device** (Raspberry Pi / NVIDIA Jetson) connected to signal relays

#### 3. Add a Safety Layer (Critical)
Real-world deployment **requires**:
- ⏱️ **Minimum green times** — pedestrians need 15+ seconds to cross
- 🔴 **All-red clearance intervals** between every phase change
- 🔄 **Failsafe fallback** — revert to fixed-time mode if the agent crashes
- 👤 **Human override** — manual control capability at all times

### Practical Next Steps

| Approach | Difficulty | Description |
|---|---|---|
| **Data-Driven Simulation** | ⭐ Easy | Feed real traffic data from Google Maps API into SUMO instead of random vehicles |
| **Digital Twin** | ⭐⭐ Medium | Build a SUMO replica of a real intersection, calibrate with sensor data, deploy learned policy |
| **Hardware-in-the-Loop** | ⭐⭐⭐ Hard | Connect the RL agent to a real traffic controller in a lab where the controller interfaces with SUMO |
| **Full Deployment** | ⭐⭐⭐⭐ Expert | Sensor integration + controller hardware + safety certification + municipal government approval |

> **Note:** The current simulation-based approach is exactly how research labs and companies like Google DeepMind and Siemens develop traffic AI before real-world deployment.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch — `git checkout -b feature/my-feature`
3. Commit your changes — `git commit -m 'Add my feature'`
4. Push to the branch — `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with** SUMO · Stable-Baselines3 · PettingZoo · PyTorch

</div>
