# DQN-PONG

A Deep Q-Network (DQN) agent implemented in PyTorch to play Atari Pong.

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/squeakyhobo/DQN-PONG.git
    cd DQN-PONG
    ```
2.  (Optional but Recommended) Create and activate a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Agent

To run the Pong agent (either for training or evaluation, depending on settings in `src/pong.py`):

```bash
python src/pong.py
```

## Trained Models

Trained model weights are saved in the `weights/` directory (ignored by Git).
