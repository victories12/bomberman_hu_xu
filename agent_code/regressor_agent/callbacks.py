import os
from collections import deque
import random

import numpy as np
import pickle
from sklearn.linear_model import SGDRegressor


from .agent_setting import ACTIONS, TRAINING_ROUNDS, EPSILON_END, EPSILON_START
from .agent_features import state_to_features,is_dangerous_to_objects, is_safe_to_drop_bomb


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()

    self.current_round = 0
    
    self.model_path = f"model/my-saved-model-{TRAINING_ROUNDS}.pt"
    if os.path.isfile(self.model_path):
        self.logger.info("Loading model from saved state.")
        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)
        self.model_is_fitted = {action:True for action in ACTIONS}

        self.epsilon = EPSILON_END # 0.01
    
    if self.train or not os.path.isfile(self.model_path):
        self.logger.info("Setting up model from scratch.")
        self.model = {action: SGDRegressor(alpha=0.0001, warm_start=True) for action in ACTIONS}
        self.model_is_fitted = {action:False for action in ACTIONS}

        self.epsilon = EPSILON_START # 0.2
    

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 7)


def act(self, game_state):
    """
    Picking action according to model
    Args:
        self:
        game_state:

    Returns:
    """
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
   
    valid_actions= get_valid_actions(game_state)

    if random.random() < self.epsilon or not all(self.model_is_fitted.values()):
        execute_action = np.random.choice(valid_actions)
    else:
        q_values = {action:self.model[action].predict(state_to_features(game_state, self.coordinate_history).reshape(1,-1))[0] for action in valid_actions}
        execute_action = max(q_values,key=q_values.get)

    _, _, bombs_left, (x, y) = game_state['self']
    self.coordinate_history.append((x, y))

    return execute_action

def get_valid_actions(game_state):
    _, _, bombs_left, (x, y) = game_state['self']
    arena = game_state['field']
    bombs = [xy for (xy, t) in game_state['bombs']]
    others = [xy for (n, s, b, xy) in game_state['others']]
    bomb_map = game_state['explosion_map']

    directions = {'UP':(0, -1), 'RIGHT':(1, 0), 'DOWN':(0, 1), 'LEFT':(-1, 0), 'WAIT':(0, 0)}

    valid_actions = []
    danger_status = np.zeros(len(directions))

    for i, (dx, dy) in enumerate(directions.values()):
        xx,yy = x+dx,y+dy
        if arena[xx,yy]==0:
            danger_status[i] = int(is_dangerous_to_objects(xx, yy, arena, bombs))
        else:
            danger_status[i] = -1

    if not any(danger_status == 0):
        danger_status = np.zeros(len(directions))
        danger_status[-1] = 1

    for i, (dx, dy) in enumerate(directions.values()):
        d = (x+dx,y+dy)
        if (arena[d] == 0 and
                bomb_map[d] < 1 and
                not d in others and
                not d in bombs and
                danger_status[i] == 0):
            valid_actions.append(ACTIONS[i])

    if bombs_left and is_safe_to_drop_bomb(x,y,arena):
        valid_actions.append(ACTIONS[-1])

    if len(valid_actions) == 0:
        return ACTIONS
    else:
        return valid_actions