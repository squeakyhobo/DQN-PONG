import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing,FrameStackObservation
import os
from Qnet import Qnet
from ReplayBuffer import ReplayBuffer
import torch
import random
import numpy as np
from collections import deque


NUM_ACTIONS = 3
EVALUATE =True

#Hyperparemeters 
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
DEVICE = torch.device("mps")



EPS_DECAY = 100000 # How many steps to go from 1.0 to 0.01
env = gym.make("ALE/Pong-v5",frameskip=1)
TARGET_UPDATE = 1000




# make pong env with wrappers to help frame stack of 4 and preprocess the obs
gym.register_envs(ale_py)

env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4,scale_obs=True)
env = FrameStackObservation(env, stack_size=4)

#initalise models
saved_weights = torch.load("weights/pong.pth")
policy_net = Qnet(NUM_ACTIONS).to(DEVICE)
policy_net.load_state_dict(saved_weights)
target_net = Qnet(NUM_ACTIONS).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
huber_loss = torch.nn.HuberLoss()

memory = ReplayBuffer(100000)
ACTIONS = [0,2,3]






def main():
    steps_done =0
    episode_reward =0
    
    q_val_mean = deque(maxlen=300)
    obs, _ = env.reset()

    for _ in range(500000):
        
        epsilon = max(EPS_END, EPS_START - steps_done / EPS_DECAY)
        
        if random.random() < epsilon:
            action = random.choice([0,1,2])
        else:
            with torch.no_grad():
               # print("optimal action used")
                # [1, 4, 84, 84] -> Move to MPS -> Get best action
                state_tensor = torch.from_numpy(np.array(obs)).unsqueeze(0).to(DEVICE)
                q_vals = policy_net(state_tensor) # Exploit
               # print(q_vals.shape)
                action = q_vals.argmax(1).item()
                highest_q_val = q_vals.max(1)[0].item()
                
                
                q_val_mean.append(highest_q_val)
                
                

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        next_obs, reward, terminated, truncated, _ = env.step(ACTIONS[action])
        done = terminated or truncated
        episode_reward+= reward
        
        memory.push(obs,action,reward,next_obs,done)
        obs = next_obs
        steps_done += 1

        # If the episode has ended then we can reset to start a new episode
        if done:
            print(f"the total reward of this episode is {episode_reward}. The epsilon is {epsilon}. The step is {steps_done} average used Q val is {np.mean(q_val_mean)}")
            episode_reward =0
            
            obs, _ = env.reset()
        
        if len(memory) > BATCH_SIZE:
            #print("training....")
            train_step()
        
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), os.path.join("weights", "pong.pth"))
            
        if steps_done% (50000)==0:      
            print("saving weights")
        
            

    env.close()

def train_step():
    obs,actions,rewards,next_obs,dones = memory.sample(BATCH_SIZE)

    obs = obs.to(DEVICE)
    actions = actions.to(DEVICE)
    rewards = rewards.to(DEVICE)
    next_obs = next_obs.to(DEVICE)
    dones = dones.to(DEVICE)

    current_qs = policy_net(obs).gather(1, actions)

    with torch.no_grad():
        # 1. SELECT (using Policy Net): "What's the best action for next_state?"
        next_actions = policy_net(next_obs).argmax(1, keepdim=True)
        
        # 2. EVALUATE (using Target Net): "What is the value of that action?"
        # Note: We use the FIXED target net here for stability!
        next_q_values = target_net(next_obs).gather(1, next_actions).squeeze(1)
        
        # 3. COMPUTE TARGET (The Bellman Equation)
        target_qs = rewards + (GAMMA * next_q_values * (1 - dones))
    
    
    loss = huber_loss(current_qs.squeeze(),target_qs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate():
    env = gym.make("ALE/Pong-v5",frameskip=1,render_mode ="human")
    gym.register_envs(ale_py)

    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4,scale_obs=True)
    env = FrameStackObservation(env, stack_size=4)

    obs,_ = env.reset()
    
    for _ in range(1000000):
        state_tensor = torch.from_numpy(np.array(obs)).unsqueeze(0).to(DEVICE)
        action = policy_net(state_tensor).argmax().item()
        next_obs, reward, terminated, truncated, _ = env.step(ACTIONS[action])
        if terminated or truncated:
                obs,_ = env.reset()
        else:
            obs = next_obs



if EVALUATE:
    evaluate()
else:
    main()

    