# 🏠 Dollhouse: A Reinforcement Learning Environment for Smart Thermal Control

Dollhouse is a simulation framework designed to train reinforcement learning (RL) agents for thermal comfort and energy efficiency in buildings. It combines physics-informed modeling using SINDy with stable-baselines3 to train intelligent control policies.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Dollhouse.git](https://github.com/confrence-anonymous/DollHouse.git
cd Dollhouse
```

### 2. Create a New Environment and Install Dependencies

Using `conda`:

```bash
conda create -n dollhouse 
conda activate dollhouse
pip install -r requirements.txt
```

Or using `venv`:

```bash
python -m venv dollhouse-env
source dollhouse-env/bin/activate  # On Windows: dollhouse-env\Scripts\activate
pip install -r requirements.txt
```

---

### 3. Train an RL Agent 

You can train an RL agent using a simple Python script like this:

```python
# train_agent.py
import os
from dollhouse_env import DollhouseThermalEnv
from train_sindy_model import train_sindy_model
from train_rl_agent import create_vectorized_env, train_rl_agent

# Path to CSV data used to fit the SINDy model
data_file = "path/to/your/dataset.csv"

# Train the SINDy model from data
sindy_model = train_sindy_model(file_path=data_file)

# Define environment parameters
env_params = {
    "episode_length": 2880,
    "time_step_seconds": 30,
    "heating_setpoint": 26.0,
    "cooling_setpoint": 28.0,
    "external_temp_pattern": "sine",
    "setpoint_pattern": "schedule",
    "reward_type": "balanced",
    "energy_weight": 1.0,
    "comfort_weight": 1.0,
    "use_reward_shaping": True,
    "random_start_time": True,
    "shaping_weight": 0.3,
}

# Create a directory for monitor logs
monitor_dir = "logs/monitor"
os.makedirs(monitor_dir, exist_ok=True)

# Create a vectorized environment with 4 parallel envs
vec_env = create_vectorized_env(
    sindy_model=sindy_model,
    env_params=env_params,
    n_envs=4,
    seed=0,
    monitor_dir=monitor_dir,
    vec_env_type="subproc",
    normalize=True,
)

# Train an RL agent (e.g., PPO)
model, model_path = train_rl_agent(
    vec_env=vec_env,
    n_envs=4,
    algorithm="ppo",
    total_timesteps=1_000_000,
    seed=0,
    log_dir="logs",
    wandb_project="dollhouse-demo",
    wandb_entity="your-wandb-username",
    use_wandb=True,
)

print(f"✅ Training complete. Model saved at: {model_path}")
```

Then run it from the terminal:

```bash
python train_agent.py
```

> 🔧 This script automatically:
>
> * Trains a SINDy dynamics model from your data
> * Creates vectorized environments
> * Trains an RL agent using PPO, A2C, DQN, or SAC
> * Logs metrics to Weights & Biases (optional)
> * Saves the final model and environment configuration

---



## 📊 Example CLI Training (Optional)

To run training from the command line:

```bash
python train_rl_agent.py \
  --data path/to/data.csv \
  --algorithm ppo \
  --timesteps 10000000 \
  --n-envs 4
```

---

## 📜 License

MIT License. Feel free to use, modify, and contribute!

---

## 📈 Acknowledgements

Built using [stable-baselines3](https://github.com/DLR-RM/stable-baselines3), [PySINDy](https://github.com/dynamicslab/pysindy), and [WandB](https://wandb.ai/).


