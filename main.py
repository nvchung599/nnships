import pygame
from defs import *

def run_game():

    pygame.init()
    gameDisplay = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption('my first nn game')

    running = True
    label_font = pygame.font.SysFont('monospace', DATA_FONT_SIZE)

    clock = pygame.time.Clock()
    dt = 0
    game_time = 0

    while running:

        dt = clock.tick(FPS)
        game_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.update()
        pygame.display.set_caption(" FPS: {:.1f}     TIME: {:.1f}".format(clock.get_fps(), game_time/1000))


if __name__ == "__main__":
    run_game()