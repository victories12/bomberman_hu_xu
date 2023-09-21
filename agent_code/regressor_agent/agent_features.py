import random
from queue import Queue
import numpy as np

import time


def state_to_features(game_state: dict, coordinate_history: Queue) -> np.array:
    """
        *This is not a required function, but an idea to structure your code.*

        Converts the game state to the input of your models, i.e.
        a feature vector.

        You can find out about the state of the game environment via game_state,
        which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
        what it contains.

        :param game_state:  A dictionary describing the current game board.
        :return: np.array
        """
    if game_state is None:
        return None
    
    arena = game_state['field']
    coins = game_state['coins']
    _, score, bombs_left, (x, y) = game_state['self']
    others = [xy for (n, s, b, xy) in game_state['others']]
    bombs = [xy for (xy, t) in game_state['bombs']]
    # print('self',x,y)
    # print('others', others)

    is_dangerous = is_dangerous_to_objects(x, y, arena, bombs)
  
    escape_direction = escape_bombers(x, y, arena, bombs, others)

    movement_towards_oppenents, oppenents_reachable = move_towards_oppenents(x, y, 5, arena, bombs, others)

    movement_towards_coins = move_towards_coins(x, y,coins, arena, bombs, others)

    movement_towards_crates, crates_reachable = move_towards_bomber_dropped_location(x, y, 10, arena, bombs, others)

    bomber_level = bomber_effect(x, y, arena)
    
    num_loop = min(coordinate_history.count((x, y)), 6)

    target_acquired = int((oppenents_reachable or (crates_reachable and all(movement_towards_oppenents == (0, 0))))
                          and bombs_left and not is_dangerous)
    
    features = np.concatenate(
        (is_dangerous, target_acquired, escape_direction, movement_towards_oppenents, movement_towards_coins, movement_towards_crates, bomber_level, num_loop),
        axis=None)
    
    return features.reshape(1, -1)[0]


def is_dangerous_to_objects(x, y, arena, objects):  # objects: bombs or others
    directions = {'UP':(0, -1), 'RIGHT':(1, 0), 'DOWN':(0, 1), 'LEFT':(-1, 0)}
    for (bx, by) in objects: 
        if bx == x and by == y:
            return True
        for dx,dy in directions.values():
            xx, yy = bx, by
            xx, yy = xx+dx, yy+dy
            while arena[xx,yy]!=-1 and abs(xx - bx) <= 3 and abs(yy - by) <= 3:
                if xx == x and yy == y:
                    return True
                xx, yy = xx+dx, yy+dy
    return False


def escape_bombers(x, y, arena, bombs, others):
    # Performs a breadth-first search of the reachable free tiles until a safe tile is encountered.
    directions = {'UP':(0, -1), 'RIGHT':(1, 0), 'DOWN':(0, 1), 'LEFT':(-1, 0)}
    escapable = False
    if bombs:
        frontier = [(x,y)]
        parent_dict = {(x,y): (x,y)}
        while len(frontier)>0:
            cx,cy = frontier.pop(0)
            if not is_dangerous_to_objects(cx, cy, arena, bombs):
                escapable = True
                current = cx,cy 
                break
            next_positions = [(cx+dx,cy+dy) for dx,dy in directions.values()]  
            random.shuffle(next_positions)
            for d in next_positions:
                if d not in parent_dict.values():
                    if arena[d] == 0 and d not in bombs and d not in others:
                        frontier.append(d)
                        parent_dict[d] = (cx,cy)
        if escapable:  
            while True:
                if parent_dict[current] == (x,y): 
                    break
                current = parent_dict[current]  
            
            if current != (x,y):
                return np.array((current[0]-x,current[1]-y))
    
    return np.zeros(2)


def vertical_sides(x, y, direction):
    if direction in ['UP', 'DOWN']:
        jx, jy, kx, ky = x + 1, y, x - 1, y
    elif direction in ['RIGHT', 'LEFT']:
        jx, jy, kx, ky = x, y + 1, x, y - 1
    else:
        raise ValueError("error")
    return jx, jy, kx, ky


def is_safe_to_drop_bomb(x, y, arena):
    directions = {'UP':(0, -1), 'RIGHT':(1, 0), 'DOWN':(0, 1), 'LEFT':(-1, 0)}
    for d, (dx,dy) in directions.items():
        xx, yy = x, y
        xx, yy = xx+dx, yy+dy
        while arena[xx,yy]==0:
            if abs(xx - x)>3 or abs(yy - y)>3:
                return True
            jx, jy, kx, ky = vertical_sides(xx, yy, d)
            if arena[jx,jy]==0 or arena[kx,ky]==0:
                return True
            xx, yy = xx+dx, yy+dy
    return False


def move_towards_oppenents(x, y, n, arena, bombs, others):
    directions = {'UP':(0, -1), 'RIGHT':(1, 0), 'DOWN':(0, 1), 'LEFT':(-1, 0)}
    reachable = False
    if others:
        frontier = [(0,(x,y))]
        parent_dict = {(x,y): (x,y)}
        while len(frontier)>0:
            num_step,(cx,cy) = frontier.pop(0)
            if num_step>n:
                continue
            if is_dangerous_to_objects(cx, cy, arena, others) and is_safe_to_drop_bomb(cx, cy, arena):
                reachable = True
                current = (cx,cy)
                break
            next_positions = [(cx+dx,cy+dy) for dx,dy in directions.values()]  
            random.shuffle(next_positions)
            for d in next_positions:
                if d not in parent_dict.values():
                    if arena[d] == 0 and d not in bombs and d not in others:
                        frontier.append((num_step+1,d))
                        parent_dict[d] = (cx,cy)
        if reachable:             
            while True:
                if parent_dict[current] == (x,y):   
                    break  
                current = parent_dict[current]  

            if current == (x,y):
                return np.zeros(2),True
            else:
                if not is_dangerous_to_objects(current[0],current[1], arena, bombs):  
                    return np.array([current[0]-x,current[1]-y]),False
    
    return np.zeros(2),False


def move_towards_coins(x, y, coins, arena, bombs, others):
    directions = {'UP':(0, -1), 'RIGHT':(1, 0), 'DOWN':(0, 1), 'LEFT':(-1, 0)}
    reachable = False
    if coins:
        frontier = [(x,y)]
        parent_dict = {(x,y): (x,y)}
        while len(frontier)>0:
            cx,cy = frontier.pop(0)
            if (cx,cy) in coins:
                reachable = True
                current = (cx,cy)
                break
            next_positions = [(cx+dx,cy+dy) for dx,dy in directions.values()]  
            random.shuffle(next_positions)
            for d in next_positions:
                if d not in parent_dict.values():
                    if arena[d] == 0 and not d in bombs and d not in others:
                        frontier.append(d)
                        parent_dict[d] = (cx,cy)
        if reachable:             
            while True:
                if parent_dict[current] == (x,y):   
                    break
                current = parent_dict[current]  

            if current != (x,y) and not is_dangerous_to_objects(current[0],current[1], arena, bombs):
                return np.array([current[0]-x,current[1]-y])
                
    return np.zeros(2)


def calculate_surrounding_crates(x, y, arena):
    num_crates = 0
    directions = {'UP':(0, -1), 'RIGHT':(1, 0), 'DOWN':(0, 1), 'LEFT':(-1, 0)}
    for d, (dx,dy) in directions.items():
        xx, yy = x, y
        xx, yy = xx+dx, yy+dy
        while arena[xx,yy]!=-1 and abs(xx - x) <= 3 and abs(yy - y) <= 3:
            if arena[xx,yy]==1:
                num_crates += 1
            xx, yy = xx+dx, yy+dy
    return num_crates


def move_towards_bomber_dropped_location(x, y, n, arena, bombs, others):
    directions = {'UP':(0, -1), 'RIGHT':(1, 0), 'DOWN':(0, 1), 'LEFT':(-1, 0)}
    crates = []
    frontier = [(0,(x,y))]
    parent_dict = {(x,y): (x,y)}
    while len(frontier)>0:
        current = frontier.pop(0)
        num_step,(cx,cy) = current
        if num_step>n:
            continue
        num_crates = calculate_surrounding_crates(cx,cy,arena)
        if num_crates>0 and is_safe_to_drop_bomb(cx,cy,arena):
            crates.append((num_crates, num_step, (cx, cy)))
        next_positions = [(cx+dx,cy+dy) for dx,dy in directions.values()]  
        random.shuffle(next_positions)
        for d in next_positions:
            if d not in parent_dict.values():
                if arena[d] == 0 and d not in bombs and d not in others:
                    frontier.append((num_step+1,d))
                    parent_dict[d] = (cx,cy)

    if len(crates)>0:
        value_max = 0
        for num_crates, num_step, (ix, iy) in crates:
            value = num_crates / (4 + num_step)
            if value > value_max:
                value_max = value
                best_node = ix, iy
                
        current = best_node
        while True:
            if parent_dict[current] == (x,y):   
                break  
            current = parent_dict[current]  

        if current == (x,y):
            return np.zeros(2),True
        else:
            if not is_dangerous_to_objects(current[0],current[1], arena, bombs): 
                return np.array([current[0]-x,current[1]-y]),False
    
    return np.zeros(2),False


def bomber_effect(x, y, arena):
    level = 0
    num_crates = calculate_surrounding_crates(x, y, arena)
    if 0 < num_crates <=2 and is_safe_to_drop_bomb(x, y, arena):
        level = 1
    elif 2 < num_crates <=5 and is_safe_to_drop_bomb(x, y, arena):
        level = 2
    elif num_crates > 5 and is_safe_to_drop_bomb(x, y, arena):
        level = 3

    return level