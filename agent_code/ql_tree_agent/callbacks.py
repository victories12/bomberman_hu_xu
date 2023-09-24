import os
import pickle
import random
import numpy as np
import sys
from collections import deque
import time
from sklearn import tree

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
FEATURES = {
    'nearest safe spot ': 'dir',
    'nearest coin ': 'dir',
    'nearest crate ': 'dir',
    'safe to bomb': 'bool',
    'enemy is trapped': 'bool',
    'bomb available': 'bool'
}
PATH = {
    0: 'not available',
    1: 'current field',
    2: 'down',
    3: 'up',
    4: 'right',
    5: 'left'
}

def find_action(model, features):
    '''
    to find the action of this array of features
    '''
    current = model
    for f in features:
        current = current[f]
    return current

def setup(self):
    '''
    function for setting up 
    define the model which store the features and action for the agent robot
    define the pt document to store the model data
    '''

    if not os.path.isfile("model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.zeros((6, 6, 6, 2, 2, 2, 6))

    elif self.train:
        self.logger.info("Loading model form saved state.")
        with open("model.pt", "rb") as file:
            self.model = pickle.load(file)
    else:
        self.logger.info("Loading model form saved state.")
        with open("model.pt", "rb") as file:
            self.model = pickle.load(file)
        self.regressor = decision_tree_regressor(self)

         
def act(self, game_state: dict) -> str:
    '''
    if train use q-learning to store the model state
    if not train use regression to predict action
    '''

    if not game_state:
        return 'WAIT'

    random_prob=.2

    features = get_features(game_state)

    if self.train and random.random() < random_prob:
        return np.random.choice(ACTIONS)

    if self.train:
    #    return np.random.choice(ACTIONS)
        return ACTIONS[np.argmax(find_action(self.model, features))]
    else:
        return predict_action(features, self.regressor)

def get_features(game_state: dict):
    '''
    to append direaction for each dimension feature 
    '''
    features = []

    _, _, bomb_available, agent_pos = game_state['self']
    field = game_state['field']
    coins = game_state['coins']
    pos_x, pos_y = agent_pos

    # direction to nearest safe spot
    goal = lambda x, y: safe_field(game_state, x, y) == 'SAFE'
    features.append(find_direction(game_state, field, pos_x, pos_y, goal, 'SEMI-SAFE'))

    # direction to nearest coin
    goal = lambda x, y: (x, y) in coins
    features.append(find_direction(game_state, field, pos_x, pos_y, goal, 'SAFE'))

    # direction to nearest crate
    goal = lambda x, y: (field[x, y+1] == 1 or field[x, y-1] == 1 or field[x+1, y] == 1 or field[x-1, y] == 1)
                                
    features.append(find_direction(game_state, field, pos_x, pos_y, goal, 'SAFE'))

    # safe to bomb 
    goal = lambda x, y: safe_field(game_state, x, y, pos_x, pos_y) == 'SAFE'
    features.append(int(find_direction(game_state, field, pos_x, pos_y, goal, 'SEMI-SAFE', max_len=4) != 0))

    # enemy is trapped:
    goal = lambda x, y: safe_field(game_state, x, y, pos_x, pos_y) == 'SAFE'
    enemy_is_trapped = False
    for _, _ , _, pos in game_state['others']:
        x_e, y_e = pos
        if find_direction(game_state, field, x_e, y_e, goal, 'SEMI-SAFE', max_len=4) == 0:
            enemy_is_trapped = True
            break
    features.append(int(enemy_is_trapped))

    # bomb available
    features.append(int(bomb_available))
    return features



def safe_field(game_state, pos_x, pos_y, bomb_x=None, bomb_y=None, only_custom_bomb=False):
    '''
    check if the given field is safe
    '''
    field = game_state['field']
    bombs = game_state['bombs'].copy()
    if bomb_x and bomb_y:
        bombs.append(((bomb_x, bomb_y), 3))
    if only_custom_bomb:
        bombs = [((bomb_x, bomb_y), 3)]
    explosion_map = game_state['explosion_map']
    safe = 'SAFE'

    if explosion_map[pos_x, pos_y] != 0:
        safe = 'UNSAFE'

    for (x, y), t in bombs:
        if (pos_x == x and abs(y - pos_y) <= 3):
            s = 1 if y > pos_y else -1 
            wall = False
            for d in range(s, y-pos_y, s):
                if field[x, pos_y+d] == -1:
                    wall = True
            if not wall:
                safe = 'SEMI-SAFE'

        if (pos_y == y and abs(x - pos_x) <= 3):
            s = 1 if x > pos_x else -1 
            wall = False
            for d in range(s, x-pos_x, s):
                if field[pos_x+d, y] == -1:
                    wall = True
            if not wall:
                safe = 'SEMI-SAFE'

    return safe

def point_in_list(x, y, l):
    if len(l) == 0: return False
    return np.min(np.sum(abs(np.array(l)[:, :2] - [x, y]), axis=1)) == 0

def find_direction(game_state, field, x_s, y_s, goal, path_type, max_len=np.inf):
    '''
    return direction which is then showed in the feature
    0: no path
    1: stay
    2, 3, 4, 5: direction is down, up, right, left
    '''
    accepted_path_types = None
    if path_type == 'SAFE': accepted_path_types = ['SAFE']
    if path_type == 'SEMI-SAFE':    accepted_path_types = ['SAFE', 'SEMI-SAFE']
    if path_type == 'UNSAFE':   accepted_path_types = ['SAFE', 'SEMI-SAFE', 'UNSAFE']

    player_positions = [(x, y, -1) for _, _ , _, (x, y) in game_state['others']]
    _, _, _, (x, y) = game_state['self']
    player_positions.append((x, y, -1))

    fields_visited = []
    fields_to_check = deque([[x_s, y_s, None]])
    while fields_to_check:
        x, y, i = fields_to_check.popleft()
        
        if goal(x, y):
            i_current = i
            length = 0
            while True:
                if x == x_s and y == y_s:
                    return 1
                length += 1
                if length > max_len:
                    return 0
                if x == x_s and y == y_s+1:
                    return 2
                if x == x_s and y == y_s-1:
                    return 3
                if x == x_s+1 and y == y_s:
                    return 4
                if x == x_s-1 and y == y_s:
                    return 5
                x, y, i_current = fields_visited[i_current]

        fields_visited.append([x, y, i])
        i = len(fields_visited) - 1
        
        safe = safe_field(game_state, x-1, y) in accepted_path_types
        if field[x-1, y] == 0 and not point_in_list(x-1, y, fields_visited + player_positions + list(fields_to_check)) and safe:
            fields_to_check.append([x-1, y, i])
        safe = safe_field(game_state, x+1, y) in accepted_path_types
        if field[x+1, y] == 0 and not point_in_list(x+1, y, fields_visited + player_positions + list(fields_to_check)) and safe:
            fields_to_check.append([x+1, y, i])
        safe = safe_field(game_state, x, y-1) in accepted_path_types
        if field[x, y-1] == 0 and not point_in_list(x, y-1, fields_visited + player_positions + list(fields_to_check)) and safe:
            fields_to_check.append([x, y-1, i])
        safe = safe_field(game_state, x, y+1) in accepted_path_types
        if field[x, y+1] == 0 and not point_in_list(x, y+1, fields_visited + player_positions + list(fields_to_check)) and safe:
            fields_to_check.append([x, y+1, i])

    return 0



def decision_tree_regressor(self):
    '''
    define a regressor 
    spilt the features and action, all of them are stored in the model.py
    store the features in X and store the action in y
    '''
    dims = (self.model).shape
#    print(dims)
    channel = []
    y = []
    for a in range(dims[0]):
        for b in range(dims[1]):
            for c in range(dims[2]):
                for d in range(dims[3]):
                    for e in range(dims[4]):
                        for f in range(dims[5]):
                            action = np.argmax(self.model[a, b, c, d, e ,f])
                            channel.append([a, b, c, d, e, f])
                            y.append([action])
                                
    
    X = np.stack(channel)
    y = np.stack(y)
    regressor = tree.DecisionTreeClassifier(min_samples_leaf = 1)
    regressor = regressor.fit(X, y)
    return regressor

def predict_action(features, regressor):
    '''
    predict the action
    '''
    action = regressor.predict([features])
    a = action[0]
#    print(ACTIONS[a])
    return ACTIONS[a]
