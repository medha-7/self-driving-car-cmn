import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import Actor, Critic

from metadrive import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.sensors.rgb_camera import RGBCamera

from collections import namedtuple, deque
from itertools import count
import random, math

import os

#DEBUG
import sys
from tabulate import tabulate
from metadrive.utils.draw_top_down_map import draw_top_down_map
import matplotlib.pyplot as plt

DEBUG = False
try:
    if sys.argv[1] == 'd':
        
        DEBUG = True
except:
    DEBUG = False

BATCH_SIZE = 2
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

data = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
map_config = {BaseMap.LANE_NUM:3}
training_env = MetaDriveEnv(dict(
    use_render=DEBUG,
    map="C",
    image_observation=True, 
    image_on_cuda=(device == torch.device("cuda")),
    traffic_density=0,
    num_scenarios=1000,
    start_seed=1000,
    map_config=map_config,
    random_lane_width=False,
    random_agent_model=False,
    random_lane_num=False,
    vehicle_config=dict(image_source="rgb_camera",),
    sensors=dict(rgb_camera=[RGBCamera, 224,224]),
))


# test_env = MetaDriveEnv(dict(
#     map="C",
#     discrete_action=True,
#     image_observation=True, 
#     # image_on_cuda=True,
#     num_scenarios=200,
#     start_seed=0,
#     random_lane_width=False,
#     random_agent_model=False,
#     random_lane_num=False,
#     vehicle_config=dict(image_source="rgb_camera",),
#     sensors=dict(rgb_camera=[RGBCamera, 224,224]),
# ))

def display_dictionary_as_grid(data):
    # Extract keys and values from the dictionary
    keys = list(data.keys())
    values = list(data.values())
    
    # Create a grid with keys in the first row and values in the second row
    grid = [keys, values]
    
    # Use tabulate to format the grid
    formatted_grid = tabulate(grid, headers='firstrow', tablefmt='grid', numalign='center')
    
    # Print the formatted grid
    return formatted_grid


# show start location and verify env config
# program will freeze until plt window is destroyed
if DEBUG:
    state, info = training_env.reset()

    # for some reason, metadrive creates 3 images per observation, the first two being blank images
    # refactor the image to form [channels, height, width] and remove the first two images
    # slicing magic to remove the first two imposter images from memory
    state = torch.tensor(state['image'], dtype=torch.float32, device=device).permute(3,2,0,1)[2:][0]
    print(state.shape)
    
    f, layout = plt.subplots(1, 2)
    layout[0].imshow(state.permute(1,2,0))
    layout[0].axis("off")
    layout[1].imshow(draw_top_down_map(training_env.current_map))
    layout[1].axis("off")
    plt.show()


# load models
actor = Actor().to(device)
target_actor = Actor().to(device)
target_critic = Critic().to(device)
critic = Critic().to(device)

target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())


# load optimizer, enable adaptive moment estimation
critic_optimizer = optim.Adam(target_actor.parameters(), lr=LR, amsgrad=True)
actor_optimizer = optim.Adam(target_actor.parameters(), lr=LR, amsgrad=True)


# load memory, max length of 10,000 points
memory = ReplayMemory(10000)

# counter for epsilon-greedy policy
steps_done = 0

def soft_update(target_net, source_net, tau):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# greedy episilon exporation-exploitation strategy
def epsilon_greedy_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        data["Status"] = "EXPLOITING"
        with torch.no_grad():
            # return independent values for steering and throttle
            return actor(state)[0], \
                actor(state)[1]
    else:
        data["Status"] = "EXPLORING"
        # return random values for steering and throttle if under the exploitation threshold
        return torch.tensor([[random.uniform(-1, 1)]], device=device, dtype=torch.float32), \
            torch.tensor([[random.uniform(-1, 1)]], device=device, dtype=torch.float32)



def optimize_model():
    if len(memory) < BATCH_SIZE:
            return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch. This converts batch-array 
    # of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # handle terminal states where batch.next_state is None
    # episodes with None state are pulled to 0
    mask = torch.tensor(tuple(map(lambda  s: s is not None, batch.next_state)), device=device, dtype=torch.bool).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    state_values_ = torch.stack([s for s in batch.next_state if s is not None])
    next_state_batch = torch.where(mask, state_values_, torch.zeros_like(state_values_))


    # magic to store steering and throttle in a single tensor
    action_batch = torch.tensor([[x for x in i] for i in batch.action], device=device)
    action_batch_s = action_batch[:,0].unsqueeze(1)
    action_batch_t = action_batch[:,1].unsqueeze(1)

    # batch.state contains a tuple with BATCH_SIZE 3D tensors, stack them to make a 4D tensor
    state_batch = torch.stack(batch.state)

    # regular concatonate for reward, it aint special
    reward_batch = torch.cat(batch.reward)

    # we have a continuous action space, life is hard
    with torch.no_grad():
        next_action_batch_ = target_actor(next_state_batch)
        next_action_batch_s = next_action_batch_[0]
        next_action_batch_t = next_action_batch_[1]
       

        next_state_action_s = target_critic(next_state_batch, next_action_batch_s)[0].squeeze()
        target_q_values_s = reward_batch + GAMMA * next_state_action_s
   

        next_state_action_t = target_critic(next_state_batch, next_action_batch_t)[1].squeeze()
        target_q_values_t = reward_batch + GAMMA * next_state_action_t


    state_action_values_s = critic(state_batch, action_batch_s)[0].squeeze()
    state_action_values_t = critic(state_batch, action_batch_t)[1].squeeze()

    # calculate loss independently
    critic_loss_s = F.mse_loss(state_action_values_s.float(), target_q_values_s.float())
    critic_loss_t = F.mse_loss(state_action_values_t.float(), target_q_values_t.float())

    critic_optimizer.zero_grad()
    critic_loss_s.backward()
    critic_loss_t.backward()
    critic_optimizer.step()

    actor_actions = actor(state_batch)

    actor_loss_s = -critic(state_batch, actor_actions[0])[0].mean()
    actor_loss_t = -critic(state_batch, actor_actions[1])[1].mean()

    actor_optimizer.zero_grad()
    actor_loss_s.backward(retain_graph=True)
    actor_loss_t.backward(retain_graph=True)
    actor_optimizer.step()

    data["Loss: Steering"] = actor_loss_s
    data["Loss: Throttle"] = actor_loss_t


    # Update target networks (soft update)
    soft_update(target_actor, actor, TAU)
    soft_update(target_critic, critic, TAU)



for i in range(1000):
    state, info = training_env.reset()
    state = torch.tensor(state['image'], dtype=torch.float32, device=device).permute(3,2,0,1)[2:][0]
    
    for t in count():
        action_s, action_t = epsilon_greedy_action(state)
        data["Steering action"] = action_s
        data["Throttle action"] = action_t
        observation, reward, terminated, truncated, _ = training_env.step([action_s.item(), action_t.item()])
        data["Reward"] = reward
        reward = torch.tensor([reward], device=device)
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation['image'], dtype=torch.float32, device=device).permute(3,2,0,1)[2:][0]
    
        memory.push(state, torch.tensor([action_s, action_t], device=device, dtype=torch.float32), next_state, reward)
        
        state = next_state

        # soft update is also done here
        optimize_model()
        if "Loss: Steering" in data.keys():
            writer.add_scalar('Loss_Steering/episode', data["Loss: Steering"], t)
            writer.add_scalar('Loss_Throttle/episode', data["Loss: Throttle"], t)
        data['Memory Size'] = len(memory)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(display_dictionary_as_grid(data))
        if terminated:
            break

    writer.add_scalar('Loss_Steering/global', data["Loss: Steering"], i)
    writer.add_scalar('Loss_Throttle/global', data["Loss: Throttle"], i)

torch.save(actor.state_dict(), os.path.join("models", f'rlcraft_{random.randint(100,999)}.pth'))
print('Complete')
writer.flush()
