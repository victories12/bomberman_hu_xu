from typing import List
from random import shuffle
import numpy as np
import copy

import events as e
from .agent_settings import ACTIONS



CLOSE_TO_OPPENTS = 'CLOSE_TO_OPPENTS'
AWAY_FROM_OPPENTS = 'AWAY_FROM_OPPENTS'
CLOSE_TO_COINS = 'CLOSE_TO_COINS'
AWAY_FROM_COINS = 'AWAY_FROM_COINS'
CLOSE_TO_CRATES = 'CLOSE_TO_CRATES'
AWAY_TO_CRATES = 'AWAY_TO_CRATES'

AWAY_FROM_BOMBER = 'AWAY_FROM_BOMBER'
CLOSE_TO_BOMBER = 'CLOSE_TO_BOMBER'

DANGER_WAIT = 'DANGER_WAIT'
JUMP_INTO_EXPLOSION = 'JUMP_INTO_EXPLOSION'

INVALID_WAIT = 'INVALID_WAIT'
VALID_WAIT = 'VALID_WAIT'

CRATES_1_TO_2_AROUND_BOMB = 'CRATES_1_TO_2_AROUND_BOMB'
CRATES_3_TO_5_AROUND_BOMB = 'CRATES_3_TO_5_AROUND_BOMB'
CRATES_OVER_5_AROUND_BOMB = 'CRATES_OVER_5_AROUND_BOMB'

REPEATED_BOMBER = 'REPEATED_BOMBER'
GET_INTO_LOOP = 'GET_INTO_LOOP'



def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def events_happened(self, game_state: dict, action: str, default_events: List[str]):

    events = copy.deepcopy(default_events) 

    arena = game_state['field']
    coins = game_state['coins']
    _, score, bombs_left, (x, y) = game_state['self']
    others = [xy for (n, s, b, xy) in game_state['others']]
    explosion_map = game_state['explosion_map']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]

    free_space = arena == 0
    if len(crates) + len(coins) == 0:
        for o in others:
            free_space[o] = False
        
        d_opponents = look_for_targets(free_space, (x, y), others, self.logger)
        if action == d_opponents:
            events.append(CLOSE_TO_OPPENTS)
        else:
            events.append(AWAY_FROM_OPPENTS)

    d_coins = look_for_targets(free_space, (x, y), coins, self.logger)
    d_crates = look_for_targets(free_space, (x, y), crates, self.logger)
    
    if action == d_coins:
        events.append(CLOSE_TO_COINS)
    
    if action == d_crates:
        events.append(CLOSE_TO_CRATES)

    
    for (xb,yb),t in bombs:
        if xb==x and abs(yb-y)<4:
            if yb>y:
                if action=='UP': events.append(AWAY_FROM_BOMBER)
                if action=='DOWN': events.append(CLOSE_TO_BOMBER)
            if yb<y:
                if action=='UP': events.append(CLOSE_TO_BOMBER)
                if action=='DOWN': events.append(AWAY_FROM_BOMBER)

        if yb==y and abs(xb-x)<4:
            if xb>x:
                if action=='RIGHT': events.append(CLOSE_TO_BOMBER)
                if action=='LEFT': events.append(AWAY_FROM_BOMBER)
            if xb<x:
                if action=='RIGHT': events.append(AWAY_FROM_BOMBER)
                if action=='LEFT': events.append(CLOSE_TO_BOMBER)

        if xb==x and yb==y:
            if action=='WAIT': events.append(DANGER_WAIT)

     # 'UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT'
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y),(x,y)]

    if action in ACTIONS[:-1]:
        next_x,next_y = directions[ACTIONS.index(action)]
        if (0 < next_x < arena.shape[0]) and (0 < next_y < arena.shape[1]):
            if arena[next_x,next_y] == 0:
                if explosion_map[next_x,next_y] > 0:
                    events.append(JUMP_INTO_EXPLOSION)

    if action=='WAIT':
        safe_directions = np.zeros(len(ACTIONS))
        for idx,d in enumerate(directions):
            if (0 < d[0] < arena.shape[0]) and (0 < d[1] < arena.shape[1]):
                if ((arena[d] == 0) and
                        (explosion_map[d] < 1) and
                        (bomb_map[d] > 0) and
                        (not d in others) and
                        (not d in bomb_xys)):
                    safe_directions[idx] = 1

        if np.sum(safe_directions[:-1]) > 0:
            events.append(INVALID_WAIT)
        else:
            if safe_directions[-1] == 1:
                events.append(VALID_WAIT)

    if action=='BOMB':
        num_crates=0
        for (i, j) in [(x + h, y) for h in range(-3, 4)] + [(x, y + h) for h in range(-3, 4)]:
            if (0 < i < arena.shape[0]) and (0 < j < arena.shape[1]):
                if arena[i,j] == 1:
                    num_crates += 1
        if num_crates<=2:
            events.append(CRATES_1_TO_2_AROUND_BOMB)
        elif num_crates<=5:
            events.append(CRATES_3_TO_5_AROUND_BOMB)
        else:
            events.append(CRATES_OVER_5_AROUND_BOMB)

        if (x,y) in self.bomb_history:
            events.append(REPEATED_BOMBER)

    if self.coordinate_history.count((x, y)) > 2:
        events.append(GET_INTO_LOOP)

    return events


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    oppent = 0.3
    coin = 0.3
    crate = 0.25
    danger = 0.4
    bombed_crate = 0.2

    game_rewards = {
        CLOSE_TO_OPPENTS:oppent,
        AWAY_FROM_OPPENTS:-oppent,
        CLOSE_TO_COINS:coin,
        # AWAY_FROM_COINS:-coin,
        CLOSE_TO_CRATES:crate,
        # AWAY_TO_CRATES:-crate,

        AWAY_FROM_BOMBER:danger,
        CLOSE_TO_BOMBER:-danger,

        DANGER_WAIT:-danger,
        JUMP_INTO_EXPLOSION:-danger,

        INVALID_WAIT:-danger/2,
        VALID_WAIT:danger,

        CRATES_1_TO_2_AROUND_BOMB:bombed_crate,
        CRATES_3_TO_5_AROUND_BOMB:bombed_crate*2,
        CRATES_OVER_5_AROUND_BOMB:bombed_crate*3,

        REPEATED_BOMBER:-0.25,
        GET_INTO_LOOP:-0.3,

       # DEFAULT EVENTS
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
        e.INVALID_ACTION: -0.5,

        # bombing
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,

        # crates, coins
        e.CRATE_DESTROYED: 0.2,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 2.,

        # kills
        e.KILLED_OPPONENT: 5.,
        e.KILLED_SELF: -10.,
        e.GOT_KILLED: -10.,
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
