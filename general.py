import random
import math
import numpy as np
from defs import *

def get_angle(player, target):

    """returns angle between ship orientation and vector to target, {0, 180 degrees}"""
    player_unit_vector = [-math.sin(math.radians(player.angle)), -math.cos(math.radians(player.angle))]
    player_to_target_vector = target.pos - player.pos
    angle_rad = math.acos(np.dot(player_unit_vector, player_to_target_vector)/(np.linalg.norm(player_unit_vector) * np.linalg.norm(player_to_target_vector)))
    angle_deg = math.degrees(angle_rad)
    return(angle_deg)

def update_label(data, title, font, x, y, game_display):
    label = font.render('{} {}'.format(title, data), 1, DATA_FONT_COLOR)
    game_display.blit(label, (x, y))
    return y

def update_data_labels(game_display, dt, game_time, num_iterations, num_alive, font):
    y_pos = 10
    gap = 20
    x_pos = 10
    y_pos = update_label(round(1000/dt,2), 'FPS', font, x_pos, y_pos + gap, game_display)
    y_pos = update_label(round(game_time/1000,2),'Game time', font, x_pos, y_pos + gap, game_display)
    y_pos = update_label(num_iterations,'Generation', font, x_pos, y_pos + gap, game_display)
#    y_pos = update_label(num_alive,'Alive', font, x_pos, y_pos + gap, game_display)

def modify_matrix(a):
    """randomly mutates a subset of elements of a 2d matrix (such as a Theta) into {-0.5, 0.5} values"""
    for x in np.nditer(a, op_flags=['readwrite']):
        if random.random() < MUTATION_WEIGHT_MODIFY_CHANCE:
            x[...] = np.random.random_sample() - 0.5

def get_mix_from_matrices(a, b):

    total_entries = a.size
    num_rows = a.shape[0]
    num_cols = a.shape[1]

    num_to_take = total_entries - int(total_entries * MUTATION_MATRIX_MIX_PERC)
    idx = np.random.choice(np.arange(total_entries), num_to_take, replace=False)

    mix_mat = np.zeros((num_rows, num_cols))

    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            if index in idx:
                mix_mat[row][col] = a[row][col]
            else:
                mix_mat[row][col] = b[row][col]

    return mix_mat
