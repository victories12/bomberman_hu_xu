import os
import pickle
import random
from collections import deque
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from .agent_model import BombermanNet, state_to_features
from .agent_settings import ACTIONS,NUM_CHANNELS,NUM_ACTIONS,EPS_START,EPS_END,LEARNING_RATE,TRAINING_ROUNDS,TRANSITION_HISTORY_SIZE
from .agent_rewards import events_happened


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    self.current_round = 0
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.model_path = f"model/my-saved-model-{TRAINING_ROUNDS}.pt"
    if os.path.isfile(self.model_path):
        self.logger.info("Loading model from saved state.")
        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)
        self.model_is_fitted = True
        self.epsilon = EPS_END

    if self.train or not os.path.isfile(self.model_path):
        self.logger.info("Setting up model from scratch.")
        self.model = BombermanNet(NUM_CHANNELS, NUM_ACTIONS).to(self.device)
        self.target_model = BombermanNet(NUM_CHANNELS, NUM_ACTIONS).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.95)  # reduce lr every 1000 steps
        self.criterion = torch.nn.MSELoss()

        self.model_is_fitted = False
        self.epsilon = EPS_START


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]

    valid_actions, mask = get_valid_actions(self, game_state)
    # Exploration vs exploitation
    if random.random() < self.epsilon or not self.model_is_fitted:
        self.logger.debug("Exploration")
        execute_action = np.random.choice(ACTIONS)  
    else:
        self.logger.debug("Exploitation:Querying model for action.")
        state_channels = state_to_features(game_state).unsqueeze(0)
        q_values = self.model(state_channels)
        if len(valid_actions)>0:
            execute_action = valid_actions[torch.argmax(q_values[0,torch.tensor(mask)]).item()]
        else:
            execute_action = ACTIONS[torch.argmax(q_values).item()]

    _, score, bombs_left, (x, y) = game_state['self']
    self.coordinate_history.append((x,y))
    if execute_action == 'BOMB':
        self.bomb_history.append((x,y))

    self.logger.debug(f'location:{(x,y)}, excuted actions: {execute_action}')

    if not self.train:
        events = events_happened(self, game_state, execute_action, [])
        self.logger.debug(f'events occurrs: {events}')

    return execute_action


def get_valid_actions(self, game_state):
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    # ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
    directions = [(x, y - 1),(x + 1, y),(x, y + 1), (x - 1, y), (x, y)]
    mask = np.zeros(len(ACTIONS))
    valid_actions = []
    for idx,d in enumerate(directions):
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_actions.append(ACTIONS[idx])
            mask[idx] = 1
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: 
        valid_actions.append('BOMB')
        mask[-1] = 1

    mask = (mask==1)
    self.logger.debug(f'Valid actions: {valid_actions}')

    return valid_actions, mask