# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from os import system, name


BOARD_SIZE = 3
MIN_NUMBER_OF_LIGHTS = BOARD_SIZE*BOARD_SIZE
POPULATION_SIZE = 50000
CROSSOVER_CHANCE = 50
MUTATION_CHANCE = 30
POPULATION_TRAINING_STEPS = 100
NUMBER_OF_GAMES = 100


def cls():
    # Windows
    if name == 'nt':
        _ = system('cls')
    # Mac and Linux
    else:
        _ = system('clear')


def start_game(mode, parameters, plot_games):
    global BOARD_SIZE, MIN_NUMBER_OF_LIGHTS, POPULATION_SIZE, CROSSOVER_CHANCE, MUTATION_CHANCE, \
        POPULATION_TRAINING_STEPS
    BOARD_SIZE, MIN_NUMBER_OF_LIGHTS, POPULATION_SIZE, CROSSOVER_CHANCE, MUTATION_CHANCE, POPULATION_TRAINING_STEPS = \
        parameters
    if mode == 0:
        play_game()
    else:
        train_population(plot_last_game=plot_games)


# Gra
def play_game():
    pattern = random_lights()
    board = pattern.copy()
    curr_moves = 0

    while True:
        cls()
        print_board(board)

        if np.sum(board == 1) == 0:
            print('\nYou won! Solution was found in {} moves.\n'.format(curr_moves))
            print('Restart game by typing: rst')
            print('Get new pattern by typing: new')
            print('Quit game by typing: exit')
            print('\nWaiting for input')
            str_in = input()
            str_in = str_in.strip().replace(' ', '').upper()

            if str_in == 'EXIT':
                break
            elif str_in == 'RST':
                board = pattern.copy()
                curr_moves = 0
            elif str_in == 'NEW':
                pattern = random_lights()
                board = pattern.copy()
                curr_moves = 0
        else:
            print('\nMoves made: {}\n'.format(curr_moves))
            print('Make move by typing row and column id: ex. a1 or 1a')
            print('Restart game by typing: rst')
            print('Get new pattern by typing: new')
            print('Quit game by typing: exit')
            print('\nWaiting for input')
            str_in = input()
            str_in = str_in.strip().replace(' ', '').upper()

            if str_in == 'EXIT':
                break
            elif str_in == 'RST':
                board = pattern.copy()
                curr_moves = 0
            elif str_in == 'NEW':
                pattern = random_lights()
                board = pattern.copy()
                curr_moves = 0
            else:
                try:
                    if str_in[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        if str_in[1] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                            move_y = BOARD_SIZE - int(str_in[0])
                            move_x = ord(str_in[1]) - 65
                            if 0 <= move_x < BOARD_SIZE and 0 <= move_y < BOARD_SIZE:
                                make_move(board, move_y, move_x)
                                curr_moves += 1
                    elif str_in[0] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                        if str_in[1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                            move_y = BOARD_SIZE - int(str_in[1])
                            move_x = ord(str_in[0]) - 65
                            if 0 <= move_x < BOARD_SIZE and 0 <= move_y < BOARD_SIZE:
                                make_move(board, move_y, move_x)
                                curr_moves += 1
                except (ValueError, IndexError):
                    pass


def make_move(board, move_y, move_x):
    board[board == 0] = -1
    board[move_y, move_x] *= -1
    if move_x > 0:
        board[move_y, move_x - 1] *= -1
    if move_x < BOARD_SIZE - 1:
        board[move_y, move_x + 1] *= -1
    if move_y > 0:
        board[move_y - 1, move_x] *= -1
    if move_y < BOARD_SIZE - 1:
        board[move_y + 1, move_x] *= -1
    board[board == -1] = 0


def random_lights():
    board = np.full(BOARD_SIZE*BOARD_SIZE, 0, dtype=np.int8)
    number_of_lights = np.random.randint(MIN_NUMBER_OF_LIGHTS, BOARD_SIZE*BOARD_SIZE + 1)
    ids = np.arange(number_of_lights)
    board[ids] = 1
    np.random.shuffle(board)
    board = board.reshape((BOARD_SIZE, BOARD_SIZE))
    return board


# Algorytm genetyczny
def train_population(plot_last_game):
    global POPULATION_SIZE, BOARD_SIZE
    population = np.random.randint(0, 2, (POPULATION_SIZE, BOARD_SIZE*BOARD_SIZE), dtype=np.int8)

    pattern = random_lights()

    board_matrix, pattern_vector = get_pattern_variables(pattern)
    step = 0
    max_scores = []
    avg_scores = []

    while step < POPULATION_TRAINING_STEPS:
        step += 1

        population_scores, population_boards = run_genetic_algorithm(population, board_matrix, pattern_vector)

        cls()
        print("Population {}/{}:".format(step, POPULATION_TRAINING_STEPS))
        print("Avg score: " + str(np.mean(population_scores)))
        print("Max score: " + str(np.amax(population_scores)))
        print("=================================================\n")

        best_specimen = population[np.argmax(population_scores)]
        best_board = population_boards[np.argmax(population_scores)]
        print('Best specimen: {}'.format(best_specimen))

        print("\nStarting pattern:")
        print_board(pattern)

        print("\nBest specimen moves(clicked lights):")
        best_specimen_moves = best_specimen.copy().reshape((BOARD_SIZE, BOARD_SIZE))
        best_specimen_moves[best_specimen_moves == 0] = -1
        print_board(best_specimen_moves)

        print('\nBest specimen final board:')
        print_board(best_board.reshape((BOARD_SIZE, BOARD_SIZE)))

        population = breed_population(population, population_scores)
        max_scores.append(np.amax(population_scores))
        avg_scores.append(np.mean(population_scores))

        if step == POPULATION_TRAINING_STEPS:
            if plot_last_game is True:
                line1, = plt.plot(np.arange(1, len(max_scores) + 1, 1), max_scores, 'b-')
                line2, = plt.plot(np.arange(1, len(max_scores) + 1, 1), avg_scores, 'r--')
                plt.xlabel('Population')
                plt.ylabel('Fitness (g)')
                if np.amax(max_scores) < 0:
                    plt.ylim(top=0)
                x_ticks = np.linspace(1, len(max_scores), np.minimum(10, len(max_scores)), dtype=np.int32)
                plt.xticks(x_ticks)
                plt.legend([line1, line2], ['g_max', 'g_avg'])
                try:
                    plt.savefig('plots/lights_pop_score')
                except FileNotFoundError:
                    pass
                plt.show()

            while True:
                cls()
                print("Population {}/{}:".format(step, POPULATION_TRAINING_STEPS))
                print("Avg score: " + str(np.mean(population_scores)))
                print("Max score: " + str(np.amax(population_scores)))
                print("=================================================\n")

                print('Best specimen: {}'.format(best_specimen))
                if np.sum(best_board == 1) == 0:
                    print('Genetic algorithm found solution in {} moves.'.format(np.sum(best_specimen)))
                else:
                    print('Genetic algorithm was not able to find any solutions.')

                print("\nStarting pattern:")
                print_board(pattern)

                print("\nBest specimen moves(clicked lights):")
                best_specimen_moves = best_specimen.copy().reshape((BOARD_SIZE, BOARD_SIZE))
                best_specimen_moves[best_specimen_moves == 0] = -1
                print_board(best_specimen_moves)

                print('\nBest specimen final board:')
                print_board(best_board.reshape((BOARD_SIZE, BOARD_SIZE)))

                print('\nRestart game by typing: rst')
                print('Quit game by typing: exit')
                print('\nWaiting for input')
                str_in = input()
                str_in = str_in.strip().replace(' ', '').upper()
                if str_in == 'EXIT':
                    break
                elif str_in == 'RST':
                    population = np.random.randint(0, 2, (POPULATION_SIZE, BOARD_SIZE * BOARD_SIZE), dtype=np.int8)
                    pattern = random_lights()
                    board_matrix, pattern_vector = get_pattern_variables(pattern)
                    step = 0
                    max_scores = []
                    avg_scores = []
                    break


def get_pattern_variables(pattern):
    board_matrix = np.identity(BOARD_SIZE*BOARD_SIZE, dtype=np.int8)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if row > 0:
                board_matrix[BOARD_SIZE*row + col, BOARD_SIZE*row + col - BOARD_SIZE] = 1
            if row < BOARD_SIZE - 1:
                board_matrix[BOARD_SIZE*row + col, BOARD_SIZE*row + col + BOARD_SIZE] = 1
            if col > 0:
                board_matrix[BOARD_SIZE*row + col, BOARD_SIZE*row + col - 1] = 1
            if col < BOARD_SIZE - 1:
                board_matrix[BOARD_SIZE*row + col, BOARD_SIZE*row + col + 1] = 1
    pattern_vector = pattern.copy().flatten()
    return board_matrix, pattern_vector


def run_genetic_algorithm(population, board_matrix, pattern_vector):
    population = population.reshape((POPULATION_SIZE, BOARD_SIZE*BOARD_SIZE)).astype(np.int8)
    population_scores = np.zeros(POPULATION_SIZE)

    population_boards = np.einsum('ij,kj -> ki', board_matrix, population)

    population_boards = np.mod(pattern_vector + population_boards, 2)
    board_scores = np.copy(population_boards)
    board_scores = np.sum(np.mod(board_scores, 2), axis=1)
    population_scores -= board_scores

    population_number_of_moves = np.sum(population.reshape(POPULATION_SIZE, BOARD_SIZE*BOARD_SIZE), axis=1)
    max_moves = BOARD_SIZE*BOARD_SIZE
    population_scores[board_scores == 0] += max_moves - population_number_of_moves[board_scores == 0]

    return population_scores, population_boards


def population_choose_best_move(population, boards):
    move_scores = np.zeros((POPULATION_SIZE, BOARD_SIZE, BOARD_SIZE))
    new_boards = np.zeros((POPULATION_SIZE, BOARD_SIZE + 2, BOARD_SIZE + 2))
    new_boards[:, 1:-1, 1:-1] = boards
    new_pop = np.zeros((POPULATION_SIZE, BOARD_SIZE + 2, BOARD_SIZE + 2))
    new_pop[:, 1:-1, 1:-1] = population

    move_scores += new_boards[:, 1:-1, 1:-1] * new_pop[:, 1:-1, 1:-1]
    for i in [-1, 1]:
        temp_boards = np.roll(new_boards, i, axis=1)
        temp_pop = np.roll(new_pop, i, axis=1)
        move_scores += temp_boards[:, 1:-1, 1:-1] * temp_pop[:, 1:-1, 1:-1]
        temp_boards = np.roll(new_boards, i, axis=2)
        temp_pop = np.roll(new_pop, i, axis=2)
        move_scores += temp_boards[:, 1:-1, 1:-1] * temp_pop[:, 1:-1, 1:-1]

    temp_scores = move_scores.reshape((POPULATION_SIZE, BOARD_SIZE*BOARD_SIZE))
    max_scores_ids = np.argmax(temp_scores, axis=1)
    max_scores_ids_y = max_scores_ids // BOARD_SIZE + 1
    max_scores_ids_x = max_scores_ids % BOARD_SIZE + 1
    pop_ids = np.arange(POPULATION_SIZE)
    new_boards[pop_ids, max_scores_ids_y[pop_ids], max_scores_ids_x[pop_ids]] = \
        -1 * new_boards[pop_ids, max_scores_ids_y[pop_ids], max_scores_ids_x[pop_ids]]
    for i in [-1, 1]:
        new_boards[pop_ids, max_scores_ids_y[pop_ids] + i, max_scores_ids_x[pop_ids]] = \
            -1 * new_boards[pop_ids, max_scores_ids_y[pop_ids] + i, max_scores_ids_x[pop_ids]]
        new_boards[pop_ids, max_scores_ids_y[pop_ids], max_scores_ids_x[pop_ids] + i] = \
            -1 * new_boards[pop_ids, max_scores_ids_y[pop_ids], max_scores_ids_x[pop_ids] + i]
    changed_boards = new_boards[:, 1:-1, 1:-1]
    return changed_boards


def population_make_next_move(population_move, boards, population_finished, game_step):
    # Wykonywanie ruchu
    new_boards = np.zeros((POPULATION_SIZE, BOARD_SIZE + 2, BOARD_SIZE + 2))
    new_boards[:, 1:-1, 1:-1] = boards
    moves_y = population_move // BOARD_SIZE + 1
    moves_x = population_move % BOARD_SIZE + 1
    pop_ids = np.arange(POPULATION_SIZE)
    pop_ids = pop_ids[population_finished == 0]
    new_boards[pop_ids, moves_y[pop_ids], moves_x[pop_ids]] = \
        -1 * new_boards[pop_ids, moves_y[pop_ids], moves_x[pop_ids]]
    for i in [-1, 1]:
        new_boards[pop_ids, moves_y[pop_ids] + i, moves_x[pop_ids]] *= -1
        new_boards[pop_ids, moves_y[pop_ids], moves_x[pop_ids] + i] *= -1
    changed_boards = new_boards[:, 1:-1, 1:-1]
    # Liczenie punktów
    flat_boards = changed_boards.reshape((POPULATION_SIZE, BOARD_SIZE * BOARD_SIZE))
    flat_boards[flat_boards < 0] = 0
    new_scores = np.sum(flat_boards, axis=1)
    # Sprawdzanie warunków końca
    finished_this_move = np.asarray(new_scores == 0, dtype=np.int32)
    population_finished[population_finished == 0] += finished_this_move[population_finished == 0] * \
                                                     (BOARD_SIZE*BOARD_SIZE - game_step)
    return changed_boards, population_finished


def population_make_same_move(boards, move_y, move_x):
    # Wykonywanie ruchu
    new_boards = np.zeros((boards.shape[0], BOARD_SIZE + 2, BOARD_SIZE + 2))
    new_boards[:, 1:-1, 1:-1] = boards
    pop_ids = np.arange(boards.shape[0])
    move_y += 1
    move_x += 1
    new_boards[pop_ids, move_y, move_x] = -1 * new_boards[pop_ids, move_y, move_x]
    for i in [-1, 1]:
        new_boards[pop_ids, move_y + i, move_x] *= -1
        new_boards[pop_ids, move_y, move_x + i] *= -1
    changed_boards = new_boards[:, 1:-1, 1:-1]
    return changed_boards


def breed_population(population, population_scores):
    # Sortowanie osobników względem uzyskanych punktów
    population_flat = population.reshape((POPULATION_SIZE, BOARD_SIZE*BOARD_SIZE))
    population_with_scores = np.zeros((population_flat.shape[0], population_flat.shape[1] + 1))
    population_with_scores[:, :-1] = population_flat
    population_with_scores[:, population_with_scores.shape[1] - 1] = population_scores
    population_with_scores = np.flip(population_with_scores[population_with_scores[:,
                                                            population_with_scores.shape[1] - 1].argsort()], axis=0)
    population = population_with_scores[:, :-1].reshape((POPULATION_SIZE, BOARD_SIZE*BOARD_SIZE))
    scores = population_with_scores[:, -1]
    # Wybór osobników za pomocą selekcji turniejowej
    best_specimens = select_best_specimens_tournament(population)
    ids = np.arange(best_specimens.shape[0] // 2)
    best_specimens_first = best_specimens[2 * ids]
    best_specimens_second = best_specimens[2 * ids + 1]
    # Dobieranie osobników w pary i rozmnażanie ich(jednopunktowe)
    pair_ids = np.arange(best_specimens.shape[0])
    np.random.shuffle(pair_ids)
    new_specimens_first = np.zeros_like(best_specimens_first)
    new_specimens_second = np.zeros_like(best_specimens_second)
    crossover_array = np.random.rand(best_specimens_first.shape[0], best_specimens_first.shape[1])
    crossover_array[crossover_array > (100 - CROSSOVER_CHANCE) / 100] = 1
    crossover_array[crossover_array < 1] = 0

    new_specimens_first[crossover_array == 0] = best_specimens_first[crossover_array == 0]
    new_specimens_first[crossover_array == 1] = best_specimens_second[crossover_array == 1]
    new_specimens_second[crossover_array == 0] = best_specimens_second[crossover_array == 0]
    new_specimens_second[crossover_array == 1] = best_specimens_first[crossover_array == 1]
    new_specimens = np.append(new_specimens_first, new_specimens_second, axis=0)

    # Mutowanie osobników
    mutated_specimens = mutate_new_specimens(new_specimens)
    new_population = np.append(best_specimens, mutated_specimens).reshape((POPULATION_SIZE, BOARD_SIZE*BOARD_SIZE))
    new_population = np.asarray(new_population, dtype=np.int8)
    return new_population


def select_best_specimens_half(population):
    best_specimens = population[:population.shape[0] // 2]
    return best_specimens


def select_best_specimens_tournament(population):
    pair_ids = np.arange(population.shape[0])
    np.random.shuffle(pair_ids)
    pair_ids_1 = pair_ids[np.arange(0, pair_ids.shape[0], 2)]
    pair_ids_2 = pair_ids[np.arange(1, pair_ids.shape[0], 2)]
    ids = np.arange(population.shape[0] // 2)
    best_ids = np.zeros(population.shape[0] // 2, dtype=np.int32)
    best_ids[ids] = np.minimum(pair_ids_1[ids], pair_ids_2[ids])
    best_specimens = population[best_ids]
    return best_specimens


def mutate_new_specimens(specimens):
    mutation_chance = np.random.randint(0, 100, specimens.shape)
    mutation_array = np.random.randint(0, 2, specimens.shape)
    mutation_chance[mutation_chance > MUTATION_CHANCE] = 0
    mutation_chance[mutation_chance > 0] = 1
    mutated_specimens = -1 * specimens * (mutation_chance - 1) + mutation_array * mutation_chance
    return mutated_specimens


# Rysowanie planszy
def print_board(board):
    board_str = ''
    for i in range(board.shape[0]):
        board_str += '--++'
        for j in range(board.shape[1]):
            if i == 0:
                board_str += '=====+'
            else:
                board_str += '-----+'
        board_str += '+\n  ||'
        for j in range(board.shape[1]):
            if board[i, j] == 0:
                board_str += '     |'
            elif board[i, j] == 1:
                board_str += 'ooooo|'
            else:
                board_str += '     |'
        board_str += '|\n{} ||'.format(BOARD_SIZE - i)
        for j in range(board.shape[1]):
            if board[i, j] == 0:
                board_str += '     |'
            elif board[i, j] == 1:
                board_str += 'ooooo|'
            else:
                board_str += '     |'
        board_str += '|\n  ||'
        for j in range(board.shape[1]):
            if board[i, j] == 0:
                board_str += '     |'
            elif board[i, j] == 1:
                board_str += 'ooooo|'
            else:
                board_str += '     |'
        board_str += '|\n'
    board_str += '--++'
    for j in range(board.shape[1]):
        board_str += '=====+'
    board_str += '+\n   |'
    for j in range(board.shape[1]):
        board_str += '  {}  |'.format(chr(65 + j))  # A, B, C...
    print(board_str)

