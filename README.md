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
import os
import time
import torch
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from train_sindy_model import train_sindy_model
from dollhouse_env import DollhouseThermalEnv


def make_env(seed, sindy_model, env_params):
    def _init():
        params = env_params.copy()
        params["random_seed"] = seed
        params["sindy_model"] = sindy_model
        env = DollhouseThermalEnv(**params)
        env = Monitor(env)
        return env
    return _init


def main(
    data_file,
    output_dir="results",
    total_timesteps=1_000_000,
    seed=0,
):
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"ppo_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Train SINDy model from CSV
    print(f"Training SINDy model from {data_file}...")
    sindy_model = train_sindy_model(file_path=data_file)

    # Define env parameters
    env_params = {
        "episode_length": 2880,  # 24 hours
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

    # Create environment (single env wrapped in DummyVecEnv)
    env = DummyVecEnv([make_env(seed, sindy_model, env_params)])

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=os.path.join(output_dir, "tensorboard"),
    )

    # Train
    print(f"Training PPO for {total_timesteps} timesteps...")
    start = time.time()
    model.learn(total_timesteps=total_timesteps)
    duration = time.time() - start
    print(f"Training finished in {duration:.2f} seconds")

    # Save model
    model_path = os.path.join(output_dir, "ppo_dollhouse_model")
    model.save(model_path)
    print(f"Model saved at: {model_path}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO on Dollhouse environment")
    parser.add_argument("--data", type=str, required=True, help="CSV file path for SINDy training")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    main(
        data_file=args.data,
        output_dir=args.output,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )

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


