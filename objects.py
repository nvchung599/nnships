from defs import *
import pygame
import numpy as np
import random
import os
from nnet import *

class Player(object):

    def __init__(self, game_display, alliance, x, y):
        self.image = pygame.image.load(os.path.join('images', PLAYER_IMAGE_FILE)).convert_alpha()
        self.game_display = game_display
        self.rect = self.image.get_rect()
        self.angle = random.random() * 360
        self.pos = np.array([x, y])
        self.alliance = alliance
        self.fitness = 0
        self.net = Net(NUM_INPUTS, NUM_HIDDEN_1, NUM_HIDDEN_2, NUM_OUTPUTS)
        self.image_ch = pygame.image.load(os.path.join('images', 'crosshair.png')).convert_alpha()

    def draw(self):
        if DISPLAY_GRAPHICS_BOOL:
            rotated = pygame.transform.rotate(self.image_ch, self.angle)
            rot_rect = rotated.get_rect()
            self.game_display.blit(rotated, (self.pos[0]-rot_rect.center[0], self.pos[1]-rot_rect.center[1]))
#            rotated = pygame.transform.rotate(self.image, self.angle)
#            rot_rect = rotated.get_rect()
#            self.game_display.blit(rotated, (self.pos[0]-rot_rect.center[0], self.pos[1]-rot_rect.center[1]))

    def rotate(self, dt, direction):
        self.angle += dt * PLAYER_ROTATE_RATE * direction
        if self.angle < 0:
            self.angle += 360
        elif self.angle > 360:
            self.angle -= 360

    def update_manual(self, dt, keys):
        if keys[pygame.K_LEFT]:
            self.rotate(dt, COUNTER_CLOCKWISE)
        if keys[pygame.K_RIGHT]:
            self.rotate(dt, CLOCKWISE)
        self.draw()

#    def update_auto(self, dt, target):
#        self.act(dt, target)
#        self.draw()
#
#    def act(self, dt, target):
#        """based on inputs. add get_inputs inline. based on outputs, act/rotate"""
#        inputs = self.get_inputs(target)
#        outputs = self.net.feed_fwd(inputs)
#        if outputs[0] > 0.5:
#            self.rotate(dt, COUNTER_CLOCKWISE)
#        if outputs[1] > 0.5:
#            self.rotate(dt, CLOCKWISE)

    def update_auto(self, dt, target):
        self.act(dt, target)
        self.draw()

    def act(self, dt, action):
        """based on inputs. add get_inputs inline. based on outputs, act/rotate"""
        if action == 0:
            self.rotate(dt, COUNTER_CLOCKWISE)
        elif action == 1:
            self.rotate(dt, CLOCKWISE)
        else:
            raise ValueError()


    def get_inputs(self, target):
        x_1 = self.angle/360
        x_2 = get_angle(self, target)
#        x_2 = target.pos[0]/DISPLAY_W
#        x_3 = target.pos[1]/DISPLAY_H
        inputs = np.array([[x_1], [x_2]])
        return inputs

    def reset(self):
        self.angle = random.random() * 360
        self.fitness = 0


class Collection(object):

    def __init__(self, game_display):
        self.game_display = game_display
        self.players = []
        for i in range(GENERATION_POPULATION):
            self.players.append(Player(game_display, i+AI_ONLY_BOOL, DISPLAY_W / 2, DISPLAY_H / 2))

    def reset(self):
        for p in self.players:
            p.reset()

    def update(self, dt, target, keys):
        for player in self.players:
            if player.alliance == 0:
                player.update_manual(dt, keys)
            else:
                player.update_auto(dt, target)
            player.fitness += 1 - (get_angle(player, target) / 180)

    def evolve(self):

        self.players.sort(key=lambda x: x.fitness, reverse=True)

        self.reset()


class Target(object):

    def __init__(self, game_display):
        self.image = pygame.image.load(os.path.join('images', TARGET_IMAGE_FILE)).convert_alpha()
        self.game_display = game_display
        self.rect = self.image.get_rect()
        self.old_pos = np.array([0, 0])
        self.cur_pos = np.array([0, 0])
        self.new_pos = np.array([0, 0])
        self.travel_vector = np.array([0, 0])
        self.cooldown_timer = 0

    def draw(self):
        if DISPLAY_GRAPHICS_BOOL:
            self.game_display.blit(self.image, (self.cur_pos[0]-self.rect.center[0], self.cur_pos[1]-self.rect.center[1]))

    def update(self, dt):

        self.cooldown_timer -= dt

        if self.cooldown_timer < 0:
            self.new_pos[0] = random.random() * DISPLAY_W
            self.new_pos[1] = random.random() * DISPLAY_H
            self.old_pos = np.copy(self.cur_pos)
            self.travel_vector = self.new_pos - self.old_pos
            self.cooldown_timer = TARGET_COOLDOWN_TIME

        self.cur_pos = self.old_pos + self.travel_vector * \
                       ((TARGET_COOLDOWN_TIME - self.cooldown_timer)/TARGET_COOLDOWN_TIME)
