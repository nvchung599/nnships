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
        self.angle = IMAGE_START_ANGLE
#        self.angle = random.random() * 360
        self.pos = np.array([x, y])
        self.alliance = alliance
        self.fitness = 0
        self.net = Net(NUM_INPUTS, NUM_HIDDEN_1, NUM_HIDDEN_2, NUM_OUTPUTS)

        self.image_ch = pygame.image.load(os.path.join('images', 'crosshair.png')).convert_alpha()

    def draw(self):

        rotated = pygame.transform.rotate(self.image_ch, self.angle)
        rot_rect = rotated.get_rect()
        self.game_display.blit(rotated, (self.pos[0]-rot_rect.center[0], self.pos[1]-rot_rect.center[1]))

        rotated = pygame.transform.rotate(self.image, self.angle)
        rot_rect = rotated.get_rect()
        self.game_display.blit(rotated, (self.pos[0]-rot_rect.center[0], self.pos[1]-rot_rect.center[1]))


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

    def update_auto(self, dt, target):
        self.act(dt, target)
        self.draw()

    def act(self, dt, target):
        """based on inputs. add get_inputs inline. based on outputs, act/rotate"""
        inputs = self.get_inputs(target)
        outputs = self.net.feed_fwd(inputs)
        if outputs[0] > 0.5:
            self.rotate(dt, COUNTER_CLOCKWISE)
        if outputs[1] > 0.5:
            self.rotate(dt, CLOCKWISE)

    def get_inputs(self, target):
        x_1 = self.angle/360
        x_2 = target.pos[0]/DISPLAY_W
        x_3 = target.pos[1]/DISPLAY_H
        inputs = np.array([[x_1], [x_2], [x_3]])
        return inputs

    def reset(self):
#        self.angle = random.random() * 360
        self.angle = 0
        self.fitness = 0


class Target(object):

    def __init__(self, game_display):
        self.image = pygame.image.load(os.path.join('images', TARGET_IMAGE_FILE)).convert_alpha()
        self.game_display = game_display
        self.rect = self.image.get_rect()
        self.pos = np.array([0, 0])
        self.cooldown_timer = 0


    def draw(self):
        self.game_display.blit(self.image, (self.pos[0]-self.rect.center[0], self.pos[1]-self.rect.center[1]))

    def update(self, dt):
        self.cooldown_timer -= dt
        if self.cooldown_timer < 0:
            self.pos[0] = random.random() * DISPLAY_W
            self.pos[1] = random.random() * DISPLAY_H * 0.1 + (DISPLAY_H * 0.9)
            self.cooldown_timer = TARGET_COOLDOWN_TIME
        self.draw()


class Collection(object):

    def __init__(self, game_display):
        self.game_display = game_display
        self.players = []
        for i in range(GENERATION_POPULATION):
            self.players.append(Player(game_display, i+1, DISPLAY_W / 2, DISPLAY_H / 2))

    def reset(self):
        for p in self.players:
            p.reset()

    def update(self, dt, target, keys):
        for player in self.players:
            if player.alliance == 0:
                player.update_manual(dt, keys)
            else:
                player.update_auto(dt, target)
            fitness_increment = 1 - (get_angle(player, target) / 180)
            if fitness_increment > 0.5:
                player.fitness += 1 - (get_angle(player, target) / 180)

    def breed(self, parent_1, parent_2):
        child_player = Player(self.game_display, 0, DISPLAY_W/2, DISPLAY_H/2)
        child_player.net.create_mixed_weights(parent_1.net, parent_2.net)
        return child_player

    def evolve(self):
        self.players.sort(key=lambda x: x.fitness, reverse=True)

        print('---------------------------')
        for p in self.players:
            print(p.fitness)

        cut_off = int(len(self.players) * MUTATION_CUT_OFF)
        good_players = self.players[0:cut_off]
        bad_players = self.players[cut_off:]
        num_bad_take = int(len(self.players) * MUTATION_BAD_TO_KEEP)

        for b in bad_players:
            b.net.modify_weights()

        new_players = []

        idx_bad_take = np.random.choice(np.arange(len(bad_players)), num_bad_take, replace=False)

        for index in idx_bad_take:
            new_players.append(bad_players[index])

        new_players.extend(good_players)

        num_children_needed = len(self.players) - len(new_players)

        while len(new_players) < len(self.players):
            idx_pair_breed = np.random.choice(np.arange(len(good_players)), 2, replace=False)
            if idx_pair_breed[0] != idx_pair_breed[1]:
                child = self.breed(good_players[idx_pair_breed[0]], good_players[idx_pair_breed[1]])
#                if random.random() < MUTATION_WEIGHT_MODIFY_CHANCE:
#                    child.net.modify_weights()
                new_players.append(child)

        self.players = new_players

        self.reset()



