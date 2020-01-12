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

    """PROTOTYPE"""
    target = Target(game_display)
    collection = Collection(game_display)
    """PROTOTYPE"""

    while running:

        dt = clock.tick(FPS)
        game_time += dt
        iteration_timer -= dt
        game_display.blit(bg_img, (0,0))
        update_data_labels(game_display, dt, game_time, generation, 0, label_font)
        keys = pygame.key.get_pressed()


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if iteration_timer < 0:
            iteration_timer = ITERATION_TIME
            collection.evolve()
            generation += 1

        collection.update(dt, target, keys)
        target.update(dt)

        pygame.display.update()


if __name__ == "__main__":
    run_game()


