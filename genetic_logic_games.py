# -*- coding: utf-8 -*-

import os
from os import system, name
import numpy as np
import mastermind
import lights_out
import connect_four


# Mastermind
MASTERMIND_TILE_COLORS = 6
MASTERMIND_TILES_IN_ROW = 4
MASTERMIND_MAX_GAME_STEPS = 15
MASTERMIND_NUMBER_OF_GAMES = 1
MASTERMIND_POP_SIZE = 1000
MASTERMIND_CROSSOVER_CHANCE = 50
MASTERMIND_MUTATION_CHANCE = 20
MASTERMIND_PLOT_LAST_GAME = False


# Lights Out
LIGHTS_BOARD_SIZE = 3
LIGHTS_MIN_LIGHTS = 9
LIGHTS_POP_SIZE = 5000
LIGHTS_CROSSOVER_CHANCE = 50
LIGHTS_MUTATION_CHANCE = 20
LIGHTS_TRAINING_STEPS = 500
LIGHTS_PLOT_LAST_GAME = False


# Connect Four
CONNECT_P1_TYPE = 0
CONNECT_P2_TYPE = 1
CONNECT_POP_SIZE = 5000
CONNECT_NUMBER_OF_STEPS = 150
CONNECT_GAMES_PER_STEP = 3
CONNECT_MOVES_PER_GAME = 5
CONNECT_CROSSOVER_CHANCE = 50
CONNECT_MUTATION_CHANCE = 15
CONNECT_PLOT_MOVE = False


def cls():
    # Windows
    if name == 'nt':
        _ = system('cls')
    # Mac and Linux
    else:
        _ = system('clear')


def main_menu():
    while True:
        cls()
        print('MAIN MENU')
        print('====================')
        print('1. Mastermind')
        print('2. Lights Out')
        print('3. Connect Four')
        print('4. Exit \n')
        print('Waiting for option input')
        str_in = input().strip()
        if str_in == '1':
            mastermind_menu()
        elif str_in == '2':
            lights_menu()
        elif str_in == '3':
            connect4_menu()
        elif str_in == '4':
            print('\n')
            break


# Mastermind
def mastermind_menu():
    while True:
        cls()
        print('MASTERMIND MENU')
        print('====================')
        print('1. Game options')
        print('2. Genetic Algorithm options')
        print('3. Play game')
        print('4. Solve with genetic algorithm')
        print('5. Exit to main menu \n')
        print('Waiting for option input')
        str_in = input().strip()
        if str_in == '1':
            mastermind_game_options()
        elif str_in == '2':
            mastermind_ga_options()
        elif str_in == '3':
            parameters = [MASTERMIND_TILE_COLORS, MASTERMIND_TILES_IN_ROW, MASTERMIND_POP_SIZE,
                          MASTERMIND_CROSSOVER_CHANCE, MASTERMIND_MUTATION_CHANCE, MASTERMIND_MAX_GAME_STEPS,
                          MASTERMIND_NUMBER_OF_GAMES]
            mastermind.start_game(mode=0, parameters=parameters, plot_games=False)
        elif str_in == '4':
            parameters = [MASTERMIND_TILE_COLORS, MASTERMIND_TILES_IN_ROW, MASTERMIND_POP_SIZE,
                          MASTERMIND_CROSSOVER_CHANCE, MASTERMIND_MUTATION_CHANCE, MASTERMIND_MAX_GAME_STEPS,
                          MASTERMIND_NUMBER_OF_GAMES]
            mastermind.start_game(mode=1, parameters=parameters, plot_games=MASTERMIND_PLOT_LAST_GAME)
        elif str_in == '5':
            break


def mastermind_game_options():
    global MASTERMIND_TILES_IN_ROW, MASTERMIND_TILE_COLORS, MASTERMIND_MAX_GAME_STEPS
    while True:
        cls()
        print('MASTERMIND GAME OPTIONS')
        print('====================')
        print('1. Number of tiles in combination - {}'.format(MASTERMIND_TILES_IN_ROW))
        print('2. Number of possible colors - {}'.format(MASTERMIND_TILE_COLORS))
        print('3. Max guesses - {}'.format(MASTERMIND_MAX_GAME_STEPS))
        print('4. Back \n')
        print('Waiting for option input')
        str_in = input().strip()
        if (str_in in ['1', '2', '3']) is True:
            print('Waiting for value')
            str_in_val = input().strip()
            try:
                value = int(str_in_val)
                if str_in == '1':
                    if value < 2:
                        value = 2
                    if value > 9:
                        value = 9
                    MASTERMIND_TILES_IN_ROW = value
                elif str_in == '2':
                    if value < 2:
                        value = 2
                    if value > 9:
                        value = 9
                    MASTERMIND_TILE_COLORS = value
                elif str_in == '3':
                    if value < 1:
                        value = 1
                    if value > 100:
                        value = 100
                    MASTERMIND_MAX_GAME_STEPS = value
            except ValueError:
                pass
        elif str_in == '4':
            print('\n')
            break


def mastermind_ga_options():
    global MASTERMIND_POP_SIZE, MASTERMIND_CROSSOVER_CHANCE, MASTERMIND_MUTATION_CHANCE, MASTERMIND_NUMBER_OF_GAMES, \
        MASTERMIND_PLOT_LAST_GAME
    while True:
        cls()
        print('MASTERMIND GENETIC ALGORITHM OPTIONS')
        print('====================')
        print('1. Population size - {}'.format(MASTERMIND_POP_SIZE))
        print('2. Crossover chance - {}%'.format(MASTERMIND_CROSSOVER_CHANCE))
        print('3. Mutation chance - {}%'.format(MASTERMIND_MUTATION_CHANCE))
        print('4. Number of games played - {}'.format(MASTERMIND_NUMBER_OF_GAMES))
        print('5. Show plot - {}'.format(MASTERMIND_PLOT_LAST_GAME))
        print('6. Back \n')
        print('Waiting for option input')
        str_in = input().strip()
        if (str_in in ['1', '2', '3', '4']) is True:
            print('Waiting for value')
            str_in_val = input().strip()
            try:
                value = int(str_in_val)
                if str_in == '1':
                    value = int(value // 4 * 4)
                    if value < 4:
                        value = 4
                    MASTERMIND_POP_SIZE = value
                elif str_in == '2':
                    if value < 0:
                        value = 0
                    if value > 100:
                        value = 100
                    MASTERMIND_CROSSOVER_CHANCE = value
                elif str_in == '3':
                    if value < 0:
                        value = 0
                    if value > 100:
                        value = 100
                    MASTERMIND_MUTATION_CHANCE = value
                elif str_in == '4':
                    if value < 1:
                        value = 1
                    MASTERMIND_NUMBER_OF_GAMES = value
            except ValueError:
                pass
        elif str_in == '5':
            if MASTERMIND_PLOT_LAST_GAME is True:
                MASTERMIND_PLOT_LAST_GAME = False
            else:
                MASTERMIND_PLOT_LAST_GAME = True
        elif str_in == '6':
            print('\n')
            break


# Lights Out
def lights_menu():
    while True:
        cls()
        print('LIGHTS OUT MENU')
        print('====================')
        print('1. Game options')
        print('2. Genetic Algorithm options')
        print('3. Play game')
        print('4. Solve with genetic algorithm')
        print('5. Exit to main menu \n')
        print('Waiting for option input')
        str_in = input().strip()
        if str_in == '1':
            lights_game_options()
        elif str_in == '2':
            lights_ga_options()
        elif str_in == '3':
            parameters = [LIGHTS_BOARD_SIZE, LIGHTS_MIN_LIGHTS, LIGHTS_POP_SIZE, LIGHTS_CROSSOVER_CHANCE,
                          LIGHTS_MUTATION_CHANCE, LIGHTS_TRAINING_STEPS]
            lights_out.start_game(mode=0, parameters=parameters, plot_games=False)
        elif str_in == '4':
            parameters = [LIGHTS_BOARD_SIZE, LIGHTS_MIN_LIGHTS, LIGHTS_POP_SIZE, LIGHTS_CROSSOVER_CHANCE,
                          LIGHTS_MUTATION_CHANCE, LIGHTS_TRAINING_STEPS]
            lights_out.start_game(mode=1, parameters=parameters, plot_games=LIGHTS_PLOT_LAST_GAME)
        elif str_in == '5':
            break


def lights_game_options():
    global LIGHTS_BOARD_SIZE, LIGHTS_MIN_LIGHTS
    while True:
        cls()
        print('LIGHTS OUT GAME OPTIONS')
        print('====================')
        print('1. Board size - {}'.format(LIGHTS_BOARD_SIZE))
        print('2. Minimum turned on lights in starting pattern - {}'.format(LIGHTS_MIN_LIGHTS))
        print('3. Back \n')
        print('Waiting for option input')
        str_in = input().strip()
        if (str_in in ['1', '2']) is True:
            print('Waiting for value')
            str_in_val = input().strip()
            try:
                value = int(str_in_val)
                if str_in == '1':
                    if value < 2:
                        value = 2
                    if value > 9:
                        value = 9
                    LIGHTS_MIN_LIGHTS = int(LIGHTS_MIN_LIGHTS * np.square(value / LIGHTS_BOARD_SIZE))
                    LIGHTS_BOARD_SIZE = value

                    if LIGHTS_MIN_LIGHTS > LIGHTS_BOARD_SIZE*LIGHTS_BOARD_SIZE:
                        LIGHTS_MIN_LIGHTS = LIGHTS_BOARD_SIZE*LIGHTS_BOARD_SIZE
                if str_in == '2':
                    if value < 1:
                        value = 1
                    if value > LIGHTS_BOARD_SIZE*LIGHTS_BOARD_SIZE:
                        value = LIGHTS_BOARD_SIZE*LIGHTS_BOARD_SIZE
                    LIGHTS_MIN_LIGHTS = value
            except ValueError:
                pass
        elif str_in == '3':
            print('\n')
            break


def lights_ga_options():
    global LIGHTS_POP_SIZE, LIGHTS_CROSSOVER_CHANCE, LIGHTS_MUTATION_CHANCE, LIGHTS_TRAINING_STEPS, \
        LIGHTS_PLOT_LAST_GAME
    while True:
        cls()
        print('LIGHTS OUT GENETIC ALGORITHM OPTIONS')
        print('====================')
        print('1. Population size - {}'.format(LIGHTS_POP_SIZE))
        print('2. Crossover chance - {}%'.format(LIGHTS_CROSSOVER_CHANCE))
        print('3. Mutation chance - {}%'.format(LIGHTS_MUTATION_CHANCE))
        print('4. Training steps - {}'.format(LIGHTS_TRAINING_STEPS))
        print('5. Show plot - {}'.format(LIGHTS_PLOT_LAST_GAME))
        print('6. Back \n')
        print('Waiting for option input')
        str_in = input().strip()
        if (str_in in ['1', '2', '3', '4']) is True:
            print('Waiting for value')
            str_in_val = input().strip()
            try:
                value = int(str_in_val)
                if str_in == '1':
                    value = int(value // 4 * 4)
                    if value < 4:
                        value = 4
                    LIGHTS_POP_SIZE = value
                elif str_in == '2':
                    if value < 0:
                        value = 0
                    if value > 100:
                        value = 100
                    LIGHTS_CROSSOVER_CHANCE = value
                elif str_in == '3':
                    if value < 0:
                        value = 0
                    if value > 100:
                        value = 100
                    LIGHTS_MUTATION_CHANCE = value
                elif str_in == '4':
                    if value < 1:
                        value = 1
                    LIGHTS_TRAINING_STEPS = value
            except ValueError:
                pass
        elif str_in == '5':
            if LIGHTS_PLOT_LAST_GAME is True:
                LIGHTS_PLOT_LAST_GAME = False
            else:
                LIGHTS_PLOT_LAST_GAME = True
        elif str_in == '6':
            print('\n')
            break


# Connect Four
def connect4_menu():
    global CONNECT_P1_TYPE, CONNECT_P2_TYPE
    while True:
        p1_string = 'player'
        if CONNECT_P1_TYPE == 1:
            p1_string = 'genetic'
        elif CONNECT_P1_TYPE == 2:
            p1_string = 'random'

        p2_string = 'player'
        if CONNECT_P2_TYPE == 1:
            p2_string = 'genetic'
        elif CONNECT_P2_TYPE == 2:
            p2_string = 'random'

        cls()
        print('CONNECT FOUR MENU')
        print('====================')
        print('1. Genetic Algorithm options')
        print('2. Change player 1(' + p1_string + ')')
        print('3. Change player 2(' + p2_string + ')')
        print('4. Play game')
        print('5. Exit to main menu \n')
        print('Waiting for option input')
        str_in = input().strip()
        if str_in == '1':
            connect4_ga_options()
        elif str_in == '2':
            CONNECT_P1_TYPE += 1
            if CONNECT_P1_TYPE > 2:
                CONNECT_P1_TYPE = 0
        elif str_in == '3':
            CONNECT_P2_TYPE += 1
            if CONNECT_P2_TYPE > 2:
                CONNECT_P2_TYPE = 0
        elif str_in == '4':
            parameters = [CONNECT_P1_TYPE, CONNECT_P2_TYPE, CONNECT_POP_SIZE, CONNECT_NUMBER_OF_STEPS, \
                         CONNECT_GAMES_PER_STEP, CONNECT_MOVES_PER_GAME, CONNECT_CROSSOVER_CHANCE, \
                         CONNECT_MUTATION_CHANCE]
            connect_four.start_game(parameters=parameters, plot_moves=CONNECT_PLOT_MOVE)
        elif str_in == '5':
            break


def connect4_ga_options():
    global CONNECT_POP_SIZE, CONNECT_CROSSOVER_CHANCE, CONNECT_MUTATION_CHANCE, CONNECT_NUMBER_OF_STEPS, \
        CONNECT_GAMES_PER_STEP, CONNECT_MOVES_PER_GAME, CONNECT_PLOT_MOVE
    while True:
        cls()
        print('CONNECT FOUR GENETIC ALGORITHM OPTIONS')
        print('====================')
        print('1. Population size - {}'.format(CONNECT_POP_SIZE))
        print('2. Crossover chance - {}%'.format(CONNECT_CROSSOVER_CHANCE))
        print('3. Mutation chance - {}%'.format(CONNECT_MUTATION_CHANCE))
        print('4. Number of steps - {}'.format(CONNECT_NUMBER_OF_STEPS))
        print('5. Number of games per step - {}'.format(CONNECT_GAMES_PER_STEP))
        print('6. Maximum number of moves per game - {}'.format(CONNECT_MOVES_PER_GAME))
        print('7. Show plot - {}'.format(CONNECT_PLOT_MOVE))
        print('8. Back \n')
        print('Waiting for option input')
        str_in = input().strip()
        if (str_in in ['1', '2', '3', '4', '5', '6']) is True:
            print('Waiting for value')
            str_in_val = input().strip()
            try:
                value = int(str_in_val)
                if str_in == '1':
                    value = int(value // 4 * 4)
                    if value < 4:
                        value = 4
                    CONNECT_POP_SIZE = value
                elif str_in == '2':
                    if value < 0:
                        value = 0
                    if value > 100:
                        value = 100
                    CONNECT_CROSSOVER_CHANCE = value
                elif str_in == '3':
                    if value < 0:
                        value = 0
                    if value > 100:
                        value = 100
                    CONNECT_MUTATION_CHANCE = value
                elif str_in == '4':
                    if value < 1:
                        value = 1
                    CONNECT_NUMBER_OF_STEPS = value
                elif str_in == '5':
                    if value < 1:
                        value = 1
                    CONNECT_GAMES_PER_STEP = value
                elif str_in == '6':
                    if value < 1:
                        value = 1
                    elif value > 21:
                        value = 21
                    CONNECT_MOVES_PER_GAME = value
            except ValueError:
                pass
        elif str_in == '7':
            if CONNECT_PLOT_MOVE is True:
                CONNECT_PLOT_MOVE = False
            else:
                CONNECT_PLOT_MOVE = True
        elif str_in == '8':
            print('\n')
            break


if __name__ == '__main__':
     os.chdir(os.path.dirname(os.path.abspath(__file__)))
     main_menu()
