import dis

import pygame
from defs import *
import math
from objects import *
from general import *


def run_game():

    pygame.init()
    running = True
    game_display = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption('NNSHIPS ROTATE ONLY')
    bg_img = pygame.image.load(BG_IMAGE_FILE)
    label_font = pygame.font.SysFont('monospace', DATA_FONT_SIZE)

    clock = pygame.time.Clock()
    dt = 0
    game_time = 0
    iteration_timer = ITERATION_TIME
    generation = 0


    """QLEARNINGQLEARNINGQLEARNINGQLEARNINGQLEARNINGQLEARNINGQLEARNINGQLEARNINGQLEARNING"""

    LOAD_MODEL = True
    SAVE_MODEL = True

    LEARNING_RATE = 0.1
    DISCOUNT = 0.9
    EPISODES = 100

    epsilon = 1
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES//20
    epsilon_decay_value = epsilon / (END_EPSILON_DECAYING-START_EPSILON_DECAYING)

    DISCRETE_OS_SIZE = [20, 20] # player.angle, target.x_pos, target.y_pos
    discrete_os_window_size = np.array([360, 360]) / np.array(DISCRETE_OS_SIZE)

    q_table = None
    if LOAD_MODEL:
        q_table = load()
    else:
        q_table = np.random.uniform(10, 20, (DISCRETE_OS_SIZE + [ACTION_SPACE_N])) # 0 = CCW, 1 = CW

    def get_discrete_state(state):
        discrete_state = np.array(state) - np.array([0,0])
        discrete_state = discrete_state / discrete_os_window_size
        return tuple(discrete_state.astype(np.int))

    """QLEARNINGQLEARNINGQLEARNINGQLEARNINGQLEARNINGQLEARNINGQLEARNINGQLEARNINGQLEARNING"""

    target = Target(game_display)
    player = Player(game_display, 1, DISPLAY_W/2, DISPLAY_H/2)

    for episode in range(EPISODES):

        if not running: break
        done = False

        current_discrete_state = get_discrete_state([player.angle, get_polar_angle(player, target)])

        # TODO SHOW_EVERY RENDER AND TIMESCALE MODE
        current_time_scale = TIME_SCALE_ACCELERATED

        while not done:

            if np.random.random() > epsilon:
                action = np.argmax(q_table[current_discrete_state])
            else:
                action = np.random.randint(0, ACTION_SPACE_N)
            dt = clock.tick(FPS)
            game_time += dt
            iteration_timer -= dt
            game_display.blit(bg_img, (0,0))
            update_data_labels(game_display, dt, game_time, generation, 0, label_font)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    running = False

            player.act(dt, action)
            target.update(dt)
            new_discrete_state = get_discrete_state([player.angle, get_polar_angle(player, target)])

            """QLEARNQLEARNQLEARNQLEARNQLEARNQLEARNQLEARNQLEARNQLEARNQLEARN"""
            if get_angle(player, target) < 15:
                reward = 1
            else:
                reward = -1
            cur_q = q_table[current_discrete_state + (action,)]
            max_q = np.max(q_table[new_discrete_state])
            updated_q = (1-LEARNING_RATE) * cur_q + LEARNING_RATE*(reward+DISCOUNT*max_q)
            q_table[current_discrete_state + (action,)] = updated_q
            current_discrete_state = new_discrete_state
            print(updated_q)
            """QLEARNQLEARNQLEARNQLEARNQLEARNQLEARNQLEARNQLEARNQLEARNQLEARN"""

            player.draw()
            target.draw()
            pygame.display.update()

            if iteration_timer < 0:
                iteration_timer = ITERATION_TIME
                done = True

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

    if SAVE_MODEL:
        save(q_table)


if __name__ == "__main__":
    run_game()


