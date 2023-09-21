from collections import namedtuple, deque
import copy

import pickle
from typing import List
import random
from itertools import groupby

import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-GUI backend, needed for plotting in non-main thread.
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1.0

import events as e
from .agent_features import state_to_features
from .agent_setting import (ACTIONS, SAVE_INTERVAL, TRAINING_ROUNDS,
                           BATCH_SIZE, TRANSITION_BUFFER_SIZE_MAX,
                           DECAY_GAMMA, N_STEP_LEARNING, N_STEP,
                           PRIORITY_LEARNING, PRIORITY_RATIO,
                           EPSILON_END, EPSILON_DECAY,
                           )

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
num_loop = 0

# Events
ESCAPABLE = "ESCAPABLE"
NOT_ESCAPABLE = "NOT_ESCAPABLE"

CRATES_1_TO_2_AROUND_BOMB = 'CRATES_1_TO_2_AROUND_BOMB'
CRATES_3_TO_5_AROUND_BOMB = 'CRATES_3_TO_5_AROUND_BOMB'
CRATES_OVER_5_AROUND_BOMB = 'CRATES_OVER_5_AROUND_BOMB'

CLOSE_TO_OPPENTS = 'CLOSE_TO_OPPENTS'
AWAY_FROM_OPPENTS = 'AWAY_FROM_OPPENTS'
CLOSE_TO_COINS = 'CLOSE_TO_COINS'
AWAY_FROM_COINS = 'AWAY_FROM_COINS'
CLOSE_TO_CRATES = 'CLOSE_TO_CRATES'
AWAY_FROM_CRATES = 'AWAY_FROM_CRATES'

VALID_WAIT = 'VALID_WAIT'
INVALID_WAIT = 'INVALID_WAIT'

DESTROY_TARGET = "DESTROY_TARGET"
MISSED_TARGET = "MISSED_TARGET"

GET_INTO_LOOP = 'GET_INTO_LOOP'


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_BUFFER_SIZE_MAX)
    self.n_step_buffer = deque(maxlen=N_STEP)
    self.is_end_round = False
    
    # Training metrics
    self.history_data ={
        'rewards':[],
        'coins':[],
        'crates':[],
        'oppents':[],
        'loss':[]
    }
    self.reward = 0
    self.coins = 0
    self.crates = 0
    self.oppents = 0
    self.epsilon_history = []
    self.losses = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if new_game_state:
        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    # Idea: Add your own events to hand out rewards
    new_events = events
    current_coordinate = None
    if self_action:
        current_coordinate = self.coordinate_history.pop()
    old_feature = state_to_features(old_game_state, self.coordinate_history)

    if current_coordinate:
        self.coordinate_history.append(current_coordinate)

    if new_game_state:
        next_feature = state_to_features(new_game_state, self.coordinate_history)
    else:
        next_feature = None

    if old_game_state and self_action:
        new_events = get_events(old_feature, self_action, events)
        reward = reward_from_events(self, new_events)
        if N_STEP_LEARNING:
            self.n_step_buffer.append(Transition(old_feature, self_action, next_feature, reward))
            if len(self.n_step_buffer) >= N_STEP:
                    n_step_reward = np.array([t.reward for t in self.n_step_buffer])
                    reward = ((DECAY_GAMMA) ** np.arange(N_STEP)).dot(n_step_reward)
                    self.transitions.append(
                        Transition(self.n_step_buffer[0][0], self.n_step_buffer[0][1], self.n_step_buffer[-1][2], reward))
        else:
            self.transitions.append(Transition(old_feature, self_action, next_feature, reward))

    training_evaluation(self,new_events,reward,self.is_end_round)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.current_round = last_game_state['round']

    self.is_end_round = True
    game_events_occurred(self, last_game_state, last_action, None, events)

    if len(self.transitions) > BATCH_SIZE:
        batch = random.sample(self.transitions, BATCH_SIZE)

        transitions_sorted = sorted(batch, key=lambda x: x.action)
        grouped_transitions = {action: list(group) for action, group in groupby(transitions_sorted, key=lambda x: x.action)}

        for action, group in grouped_transitions.items():
            X = np.array([t.state for t in group])
            y = np.array([t.reward for t in group])  
            if all(self.model_is_fitted.values()):
                # q learning
                next_reward = np.array([(np.max([self.model[aa].predict(t.next_state.reshape(1,-1))[0] for aa in ACTIONS]) if t.next_state is not None else 0) for t in group])
                y = y + DECAY_GAMMA * next_reward

                if PRIORITY_LEARNING:
                    # calculate q estimate
                    y_predict = self.model[action].predict(X)  
                    losses = y_predict - y

                    priority_size = int(len(losses) * PRIORITY_RATIO)
                    idx = np.argpartition(losses, -priority_size)

                    X_new = X[idx][-priority_size:]
                    y_new = y[idx][-priority_size:]
                    X = X_new
                    y = y_new

            self.model[action].partial_fit(X, y)
            self.model_is_fitted[action] = True
            
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
        self.logger.debug(f'epsilon is {self.epsilon}')

    self.logger.debug(f'======Round {last_game_state["round"]} ends======\n')

    # Store the model
    if self.current_round  % SAVE_INTERVAL == 0 or self.current_round  == TRAINING_ROUNDS:
        with open(f'model/my-saved-model-{self.current_round}.pt', "wb") as file:
            pickle.dump(self.model, file)
        

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    global num_loop
    
    escape = 0.6
    bombing = 0.5
    waiting = 0.5

    oppents = 0.3
    coins = 0.2
    crates = 0.05
    
    game_rewards = {
        # SPECIAL EVENTS

        ESCAPABLE: escape,
        NOT_ESCAPABLE: -escape,

        CRATES_1_TO_2_AROUND_BOMB: 0.1,
        CRATES_3_TO_5_AROUND_BOMB: 0.3,
        CRATES_OVER_5_AROUND_BOMB: 0.5,

        CLOSE_TO_OPPENTS: oppents,
        AWAY_FROM_OPPENTS: -oppents,
        CLOSE_TO_COINS: coins,
        AWAY_FROM_COINS: -coins,
        CLOSE_TO_CRATES: crates,
        AWAY_FROM_CRATES: -crates,

        VALID_WAIT: waiting,
        INVALID_WAIT: -waiting,

        DESTROY_TARGET: bombing,
        MISSED_TARGET: -4 * escape,

        GET_INTO_LOOP: -0.025 * num_loop,

        # DEFAULT EVENTS
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
        e.INVALID_ACTION: -1,

        # bombing
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,

        # crates, coins
        e.CRATE_DESTROYED: 0.4,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 2,

        # kills
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 0,

        # passive
        e.SURVIVED_ROUND: 0,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def get_events(last_game_feature, self_action, default_events)->list:
    """
    get events
    """
    global num_loop

    events = copy.deepcopy(default_events)

    is_dangerous = last_game_feature[0] == 1
    target_acquired = last_game_feature[1]
    escape_direction = last_game_feature[2:4]
    movement_towards_oppenents = last_game_feature[4:6]
    movement_towards_coins = last_game_feature[6:8]
    movement_towards_crates = last_game_feature[8:10]
    bomber_level = last_game_feature[10]
    num_loop = last_game_feature[11]

    directions = {'UP':(0, -1), 'RIGHT':(1, 0), 'DOWN':(0, 1), 'LEFT':(-1, 0)}
    if is_dangerous:
        if self_action in directions:
            if all(directions[self_action] == escape_direction):
                events.append(ESCAPABLE)
            else:
                events.append(NOT_ESCAPABLE)
    else:
        if num_loop > 2:
            events.append(GET_INTO_LOOP)

        if self_action == 'BOMB':
            if bomber_level == 1:
                events.append(CRATES_1_TO_2_AROUND_BOMB)
            elif bomber_level == 2:
                events.append(CRATES_3_TO_5_AROUND_BOMB)
            elif bomber_level == 3:
                events.append(CRATES_OVER_5_AROUND_BOMB)

            if target_acquired == 1:
                events.append(DESTROY_TARGET)
            else:
                events.append(MISSED_TARGET)

        elif self_action == 'WAIT':
            if (all(movement_towards_oppenents == (0, 0)) and all(movement_towards_coins == (0, 0)) and all(movement_towards_crates == (0, 0)) and
                    target_acquired == 0):
                events.append(VALID_WAIT)
            else:
                events.append(INVALID_WAIT)

        else:
            if target_acquired == 1:
                events.append(MISSED_TARGET)

            if not all(movement_towards_oppenents == (0, 0)):
                if all(directions[self_action] == movement_towards_oppenents):
                    events.append(CLOSE_TO_OPPENTS)
                else:
                    events.append(AWAY_FROM_OPPENTS)

            if not all(movement_towards_coins == (0, 0)):
                if all(directions[self_action] == movement_towards_coins):
                    events.append(CLOSE_TO_COINS)
                else:
                    events.append(AWAY_FROM_COINS)

            if not all(movement_towards_crates == (0, 0)):
                if all(directions[self_action] == movement_towards_crates): 
                    events.append(CLOSE_TO_CRATES)
                else:
                    events.append(AWAY_FROM_CRATES)
    return events


def training_evaluation(self,events,reward,is_end_round=False):
    
    self.reward += reward
    self.coins += events.count(e.COIN_COLLECTED)
    self.crates += events.count(e.CRATE_DESTROYED)
    self.oppents += events.count(e.KILLED_OPPONENT)

    if is_end_round:
        if reward>0:
            print('reward',self.current_round,reward)
        self.is_end_round = False
        self.history_data['rewards'].append(self.reward)
        self.history_data['coins'].append(self.coins)
        self.history_data['crates'].append(self.crates)
        self.history_data['oppents'].append(self.oppents)
        
        self.reward = 0
        self.coins = 0
        self.crates = 0
        self.oppents = 0

        if self.current_round % SAVE_INTERVAL == 0:
            fig, ax = plt.subplots(4, figsize=(12, 12))

            # Plotting reward history
            ax[0].plot(self.history_data['rewards'], label='rewards', color='blue')
            ax[0].set_title("Training rewards over Episodes")
            ax[0].set_xlabel("Episode")
            ax[0].set_ylabel("rewards")
            ax[0].legend()

            # Plotting coins history
            ax[1].plot(self.history_data['coins'], label='coins', color='blue')
            ax[1].set_title("Training: num of collected coins over Episodes")
            ax[1].set_xlabel("Episode")
            ax[1].set_ylabel("Coins")
            ax[1].legend()

            # Plotting crates history
            ax[2].plot(self.history_data['crates'], label='crates', color='blue')
            ax[2].set_title("Training: num of destroyed crates over Episodes")
            ax[2].set_xlabel("Episode")
            ax[2].set_ylabel("crates")
            ax[2].legend()

            # Plotting oppents history
            ax[3].plot(self.history_data['oppents'], label='oppents', color='blue')
            ax[3].set_title("Training:num of killed oppents over Episodes")
            ax[3].set_xlabel("Episode")
            ax[3].set_ylabel("oppents")
            ax[3].legend()

            plt.tight_layout()
            plt.savefig(f'model/train_metrics_evaluation-{TRAINING_ROUNDS}.pdf')
