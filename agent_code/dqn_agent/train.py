import random
import math
import pickle
from typing import List
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import torch


import events as e
import settings as s
from .agent_model import state_to_features
from .agent_rewards import reward_from_events,events_happened
from .agent_settings import (ACTIONS, NUM_CHANNELS, TRANSITION_HISTORY_SIZE, BATCH_SIZE,
                            GAMMA, EPS_DECAY, TRAINING_ROUNDS, SAVE_INTERVAL)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    new_events = events_happened(self, old_game_state, self_action, events)
    self.transitions.append(Transition(state_to_features(old_game_state), ACTIONS.index(self_action), 
                                       state_to_features(new_game_state), reward_from_events(self, new_events)))
    training_evaluation(self, new_events, is_end_round=False)


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
    
    # self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    new_events = events_happened(self, last_game_state, last_action, events)
    self.transitions.append(Transition(state_to_features(last_game_state), ACTIONS.index(last_action), 
                                       np.zeros((NUM_CHANNELS, s.COLS, s.ROWS)), reward_from_events(self, new_events)))
    training_evaluation(self, new_events, is_end_round=True)

    if len(self.transitions) > BATCH_SIZE:
        batch = random.sample(self.transitions, BATCH_SIZE)
        last_game_states,last_actions,new_game_states,rewards = zip(*batch)

        states = torch.tensor(np.stack(last_game_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.stack(last_actions), dtype=torch.int64)
        next_states = torch.tensor(np.stack(new_game_states), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.stack(rewards), dtype=torch.float32)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        # Double DQN
        online_actions = self.model(next_states).max(1)[1]
        next_q_values = self.target_model(next_states).gather(1, online_actions.unsqueeze(1)).squeeze()
        # next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + GAMMA * next_q_values

        loss = self.criterion(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.model_is_fitted = True

        if self.current_round % 1000 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.epsilon *= EPS_DECAY
        self.logger.info(f"Training policy e-greedy:{self.epsilon}")

        training_loss(self, loss.item(), self.epsilon)

    self.logger.debug(f'======Round {last_game_state["round"]} ends======\n')

    # Store the model
    if self.current_round  % SAVE_INTERVAL == 0 or self.current_round  == TRAINING_ROUNDS:
        with open(self.model_path, "wb") as file:
            pickle.dump(self.model, file)


def training_evaluation(self,events,is_end_round=False):
    
    self.reward += reward_from_events(self, events)
    self.coins += events.count(e.COIN_COLLECTED)
    self.crates += events.count(e.CRATE_DESTROYED)
    self.oppents += events.count(e.KILLED_OPPONENT)

    if is_end_round:
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


def training_loss(self, loss, epsilon):
    self.losses.append(loss)
    self.epsilon_history.append(epsilon)

    if self.current_round % SAVE_INTERVAL == 0:
        fig, ax = plt.subplots(2, figsize=(12, 12))

        # Plotting epsilon history
        ax[0].plot(self.losses, label='Losses', color='blue')
        ax[0].set_title("Training losses over Episodes")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("Losses")
        ax[0].legend()

        # Plotting epsilon history
        ax[1].plot(self.epsilon_history, label='Epsilon', color='blue')
        ax[1].set_title("Epsilon Decay over Episodes")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Epsilon")
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(f'model/train_losses_evaluation-{TRAINING_ROUNDS}.pdf')