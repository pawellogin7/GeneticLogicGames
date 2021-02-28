# -*- coding: utf-8 -*-

from os import system, name
import numpy as np
import matplotlib.pyplot as plt

# Stałe
NUMBER_OF_COLUMNS = 7
NUMBER_OF_ROWS = 6
P1_TYPE = 0
P2_TYPE = 1
POPULATION_SIZE = 500
TRAINING_STEPS = 100
GAMES_PER_STEP = 4
MOVES_PER_GAME = 20
CROSSOVER_CHANCE = 50
MUTATION_CHANCE = 15
PLOT_MOVE = False


def cls():
    # Windows
    if name == 'nt':
        _ = system('cls')
    # Mac and Linux
    else:
        _ = system('clear')


def start_game(parameters, plot_moves):
    global P1_TYPE, P2_TYPE, POPULATION_SIZE, TRAINING_STEPS, GAMES_PER_STEP, MOVES_PER_GAME,  CROSSOVER_CHANCE, \
        MUTATION_CHANCE, PLOT_MOVE
    P1_TYPE, P2_TYPE, POPULATION_SIZE, TRAINING_STEPS, GAMES_PER_STEP, MOVES_PER_GAME, CROSSOVER_CHANCE, \
    MUTATION_CHANCE = parameters
    PLOT_MOVE = plot_moves
    play_game()


def play_game():
    board = np.zeros((NUMBER_OF_ROWS, NUMBER_OF_COLUMNS), dtype=np.int8)
    current_player = 1
    winner = 0

    while True:
        cls()
        print_board(board)
        if np.sum(board == 0) == 0:
            print("\nGame over! It's a tie. \n")
            print('Restart game by typing: rst')
            print('Quit game by typing: exit')
            str_in = input()
            str_in = str_in.strip().replace(' ', '').upper()
            if str_in == 'EXIT':
                break
            elif str_in == 'RST':
                board = np.zeros((NUMBER_OF_ROWS, NUMBER_OF_COLUMNS), dtype=np.int8)
                current_player = 1
                winner = 0
        elif winner != 0:
            print("\nGame over! Player {} has won. \n".format(winner))
            print('Restart game by typing: rst')
            print('Quit game by typing: exit')
            str_in = input()
            str_in = str_in.strip().replace(' ', '').upper()
            if str_in == 'EXIT':
                break
            elif str_in == 'RST':
                board = np.zeros((NUMBER_OF_ROWS, NUMBER_OF_COLUMNS), dtype=np.int8)
                current_player = 1
                winner = 0
        else:
            move_possible = False
            if current_player == 1:
                print('\nCurrent player: 1\n')
            else:
                print('\nCurrent player: 2\n')
            if (current_player == 1 and P1_TYPE == 0) or (current_player == -1 and P2_TYPE == 0):
                print('Make move by typing column id: ex. 3')
                print('Restart game by typing: rst')
                print('Quit game by typing: exit')
                print('\nWaiting for input')
                str_in = input()
                str_in = str_in.strip().replace(' ', '').upper()

                if str_in == 'EXIT':
                    break
                elif str_in == 'RST':
                    board = np.zeros((NUMBER_OF_ROWS, NUMBER_OF_COLUMNS), dtype=np.int8)
                    current_player = 1
                    winner = 0
                else:
                    try:
                        move = int(str_in) - 1
                        if 0 <= move < NUMBER_OF_COLUMNS:
                            move_possible = make_move(board, move, current_player)
                    except ValueError:
                        pass
            elif (current_player == 1 and P1_TYPE == 1) or (current_player == -1 and P2_TYPE == 1):
                move = int(genetic_algorithm_choose_best_move(board, current_player))
                move_possible = make_move(board, move, current_player)
            else:
                move = np.random.randint(7)
                move_possible = make_move(board, move, current_player)
            if move_possible is True:
                game_ended = check_board(board)
                if game_ended is True:
                    if current_player == 1:
                        winner = 1
                    else:
                        winner = 2
                else:
                    if current_player == 1:
                        current_player = -1
                    else:
                        current_player = 1


def make_move(board, column_id, current_player):
    column = board[:, column_id]
    empty = np.asarray(np.where(column == 0), dtype=np.int8)
    if empty.shape[1] > 0:
        row_id = empty[0, empty.shape[1] - 1]
        board[row_id, column_id] = current_player
        return True
    else:
        return False


def check_board(board):
    board_flat = board.flatten()
    # pattern = oooo
    step = 1
    ids = np.tile(np.asarray([0, 1, 2, 3]), NUMBER_OF_ROWS)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 4) * NUMBER_OF_COLUMNS
    pattern_1_sum = board_flat[ids] + board_flat[ids + step] + board_flat[ids + 2 * step] + \
                    board_flat[ids + 3 * step]
    # pattern = o
    #           o
    #           o
    #           o
    step = NUMBER_OF_COLUMNS
    ids = np.tile(np.asarray([0, 1, 2, 3, 4, 5, 6]), NUMBER_OF_ROWS - 3)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 7) * NUMBER_OF_COLUMNS
    pattern_2_sum = board_flat[ids] + board_flat[ids + step] + board_flat[ids + 2 * step] + \
                    board_flat[ids + 3 * step]
    # pattern = o
    #            o
    #             o
    #              o
    step = NUMBER_OF_COLUMNS + 1
    ids = np.tile(np.asarray([0, 1, 2, 3]), NUMBER_OF_ROWS - 3)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 4) * NUMBER_OF_COLUMNS
    pattern_3_sum = board_flat[ids] + board_flat[ids + step] + board_flat[ids + 2 * step] + \
                    board_flat[ids + 3 * step]
    # pattern =    o
    #             o
    #            o
    #           o
    step = NUMBER_OF_COLUMNS - 1
    ids = np.tile(np.asarray([3, 4, 5, 6]), NUMBER_OF_ROWS - 3)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 4) * NUMBER_OF_COLUMNS
    pattern_4_sum = board_flat[ids] + board_flat[ids + step] + board_flat[ids + 2 * step] + \
                    board_flat[ids + 3 * step]

    pattern_1_sum = np.abs(pattern_1_sum)
    pattern_2_sum = np.abs(pattern_2_sum)
    pattern_3_sum = np.abs(pattern_3_sum)
    pattern_4_sum = np.abs(pattern_4_sum)

    winning_moves = np.sum(pattern_1_sum == 4) + np.sum(pattern_2_sum == 4) + np.sum(pattern_3_sum == 4) +\
                    np.sum(pattern_4_sum == 4)
    if winning_moves > 0:
        return True
    else:
        return False


# Funkcje algorytmu genetycznego
def genetic_algorithm_choose_best_move(board, current_player):
    possible_moves = np.sum(board == 0, axis=0)
    possible_move_ids = np.arange(NUMBER_OF_COLUMNS)
    possible_moves = possible_move_ids[possible_moves > 0]

    for move in possible_moves:
        board_temp = board.copy()
        _ = make_move(board_temp, move, current_player)
        if check_board(board_temp) is True:
            return move

    for move in possible_moves:
        board_temp = board.copy()
        _ = make_move(board_temp, move, -1*current_player)
        if check_board(board_temp) is True:
            return move

    if current_player == 1:
        curr_player_temp = 1
    else:
        curr_player_temp = 2

    population_p1, population_p2 = random_population(possible_moves)

    max_move_scores = np.zeros((TRAINING_STEPS, NUMBER_OF_COLUMNS))
    avg_move_scores = np.zeros((TRAINING_STEPS, NUMBER_OF_COLUMNS))
    valid_move_scores = np.zeros((TRAINING_STEPS, NUMBER_OF_COLUMNS))

    for i in range(TRAINING_STEPS):
        p1_scores = np.zeros(POPULATION_SIZE)
        p2_scores = np.zeros(POPULATION_SIZE)
        for j in range(GAMES_PER_STEP):
            cls()
            print_board(board)
            print('\nPlayer {} choosing move with genetic algorithm...'.format(curr_player_temp))
            print('Training step {}/{}, game {}/{}'.format(i + 1, TRAINING_STEPS, j + 1, GAMES_PER_STEP))

            random_ids = np.arange(POPULATION_SIZE)
            np.random.shuffle(random_ids)
            game_p1_scores, game_p2_scores, final_boards = run_genetic_algorithm(board, population_p1,
                                                                                 population_p2[random_ids],
                                                                                 current_player)
            p1_scores += game_p1_scores
            p2_scores[random_ids] += game_p2_scores

        for mov_id in range(NUMBER_OF_COLUMNS):
            if current_player == 1:
                temp_pop = population_p1
                temp_scores = p1_scores
            else:
                temp_pop = population_p2
                temp_scores = p2_scores
            if np.sum(temp_pop[:, 0] == mov_id) > 0:
                max_move_scores[i, mov_id] = np.argmax(temp_scores[temp_pop[:, 0] == mov_id])
                avg_move_scores[i, mov_id] = np.mean(temp_scores[temp_pop[:, 0] == mov_id])
                valid_move_scores[i, mov_id] = i + 1

        population_p1 = breed_population(population_p1, p1_scores, possible_moves)
        population_p2 = breed_population(population_p2, p2_scores, possible_moves)

    new_max_move_scores = np.zeros((TRAINING_STEPS, NUMBER_OF_COLUMNS))
    new_avg_move_scores = np.zeros((TRAINING_STEPS, NUMBER_OF_COLUMNS))
    for i in range(NUMBER_OF_COLUMNS):
        if np.sum(valid_move_scores[:, i] > 0) > 0:
            for j in range(TRAINING_STEPS):
                max_score = 0
                avg_score = 0
                cor_sum = 0
                if valid_move_scores[j, i] != 0:
                    max_score = max_move_scores[j, i]
                    avg_score = avg_move_scores[j, i]
                    cor_sum = 1
                cor_count = 4
                cor_decrease = 1
                if j > 0:
                    for k in range(1, np.minimum(cor_count, j) + 1):
                        max_score += new_max_move_scores[j - k, i] * np.power(cor_decrease, k)
                        avg_score += new_avg_move_scores[j - k, i] * np.power(cor_decrease, k)
                        cor_sum += np.power(cor_decrease, k)
                max_score /= cor_sum
                avg_score /= cor_sum
                new_max_move_scores[j, i] = max_score
                new_avg_move_scores[j, i] = avg_score

    max_move_scores = new_max_move_scores
    avg_move_scores = new_avg_move_scores

    if PLOT_MOVE is True:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(14, 8)
        plot1_lines = []
        plot1_legend_labels = []
        plot2_lines = []
        plot2_legend_labels = []
        for i in range(NUMBER_OF_COLUMNS):
            if np.sum(valid_move_scores[:, i] > 0) > 0:
                x_axis = np.arange(1, TRAINING_STEPS + 1)
                max_move_score = max_move_scores[:, i]
                line1, = ax1.plot(x_axis, max_move_score)
                plot1_lines.append(line1)
                plot1_legend_labels.append('Column {}'.format(i+1))
                avg_move_score = avg_move_scores[:, i]
                line2, = ax2.plot(x_axis, avg_move_score)
                plot2_lines.append(line2)
                plot2_legend_labels.append('Column {}'.format(i + 1))
            x_ticks = np.linspace(1, TRAINING_STEPS, np.minimum(10, TRAINING_STEPS), dtype=np.int32)
            ax1.set_xticks(x_ticks)
            ax1.set_title('Max scores per move')
            ax1.legend(plot1_lines, plot1_legend_labels)
            ax1.set(xlabel='Population', ylabel='Max score')
            x_ticks = np.linspace(1, TRAINING_STEPS, np.minimum(10, TRAINING_STEPS), dtype=np.int32)
            ax2.set_xticks(x_ticks)
            ax2.set_title('Average scores per move')
            ax2.legend(plot2_lines, plot2_legend_labels)
            ax2.set(xlabel='Population', ylabel='Avg score')
        try:
            fig.savefig('plots/connect4_move_scores')
        except FileNotFoundError:
            pass
        plt.show()

    best_move = 0
    best_score = -1000000000
    for i in range(NUMBER_OF_COLUMNS):
        if np.sum(valid_move_scores[:, i] > 0) > 0:
            if avg_move_scores[len(avg_move_scores) - 1, i] > best_score:
                best_move = i
                best_score = avg_move_scores[len(avg_move_scores) - 1, i]
    return best_move


def random_population(possible_moves):
    population_p1 = np.random.choice(possible_moves, (POPULATION_SIZE, MOVES_PER_GAME))
    population_p2 = np.random.choice(possible_moves, (POPULATION_SIZE, MOVES_PER_GAME))
    return population_p1, population_p2


def run_genetic_algorithm(current_board, population_p1, population_p2, current_player):
    population_boards = np.repeat(current_board[np.newaxis, :, :], POPULATION_SIZE, axis=0)
    population_scores = np.zeros(POPULATION_SIZE)
    prev_moves_made = np.ones(POPULATION_SIZE, dtype=np.int8)
    population_finished = np.zeros(POPULATION_SIZE, dtype=np.int8)
    population_finished_move = np.full(POPULATION_SIZE, 2 * MOVES_PER_GAME, dtype=np.int8)
    for i in range(2 * MOVES_PER_GAME):
        prev_moves_made[population_finished == 1] = 0
        if current_player == 1:
            new_boards, curr_moves_made = population_make_move(population_boards[prev_moves_made == 1],
                                                               population_p1[prev_moves_made == 1, i // 2], current_player)
            new_finished = check_population_finished(new_boards, current_player)
            pop_finished_temp = population_finished_move[prev_moves_made == 1]
            pop_finished_temp[new_finished == 1] = i
            population_finished[prev_moves_made == 1] = new_finished
            current_player = -1
        else:
            new_boards, curr_moves_made = population_make_move(population_boards[prev_moves_made == 1],
                                                              population_p2[prev_moves_made == 1, i // 2],
                                                               current_player)
            new_finished = check_population_finished(new_boards, current_player)
            pop_finished_temp = population_finished_move[prev_moves_made == 1]
            pop_finished_temp[new_finished == 1] = i
            population_finished[prev_moves_made == 1] = new_finished
            current_player = 1
        curr_moves_made_temp = np.zeros(POPULATION_SIZE)
        curr_moves_made_temp[prev_moves_made == 1] = curr_moves_made
        population_boards[curr_moves_made_temp == 1] = new_boards[curr_moves_made == 1]
        curr_moves_made_temp = np.ones(POPULATION_SIZE)
        curr_moves_made_temp[prev_moves_made == 1] = curr_moves_made
        prev_moves_made = curr_moves_made_temp

    population_scores += population_evaluate_boards(population_boards, current_player)
    if current_player == 1:
        p1_scores = population_scores
        p2_scores = -1 * population_scores
    else:
        p1_scores = -1 * population_scores
        p2_scores = population_scores
    return p1_scores, p2_scores, population_boards


def population_make_move(boards, MOVES_PER_GAME, current_player):
    pop_ids = np.arange(boards.shape[0])
    column_ids = np.asarray(MOVES_PER_GAME, dtype=np.int32)
    possible_moves = np.zeros(boards.shape[0], dtype=np.int8)
    possible_moves[pop_ids] = np.sum(boards == 0, axis=1)[pop_ids, column_ids[pop_ids]]
    boards[possible_moves > 0, possible_moves[possible_moves > 0] - 1, column_ids[possible_moves > 0]] = \
        current_player
    moves_made = np.zeros(boards.shape[0])
    moves_made[possible_moves > 0] = 1
    return boards, moves_made


def check_population_finished(boards, current_player):
    boards_flat = boards.reshape((boards.shape[0], NUMBER_OF_COLUMNS*NUMBER_OF_ROWS))

    # pattern = oooo
    step = 1
    ids = np.tile(np.asarray([0, 1, 2, 3]), NUMBER_OF_ROWS)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 4) * NUMBER_OF_COLUMNS
    pattern_1_sum = boards_flat[:, ids] + boards_flat[:, ids + step] + boards_flat[:, ids + 2 * step] + \
                    boards_flat[:, ids + 3 * step]
    # pattern = o
    #           o
    #           o
    #           o
    step = NUMBER_OF_COLUMNS
    ids = np.tile(np.asarray([0, 1, 2, 3, 4, 5, 6]), NUMBER_OF_ROWS - 3)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 7) * NUMBER_OF_COLUMNS
    pattern_2_sum = boards_flat[:, ids] + boards_flat[:, ids + step] + boards_flat[:, ids + 2 * step] + \
                    boards_flat[:, ids + 3 * step]
    # pattern = o
    #            o
    #             o
    #              o
    step = NUMBER_OF_COLUMNS + 1
    ids = np.tile(np.asarray([0, 1, 2, 3]), NUMBER_OF_ROWS - 3)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 4) * NUMBER_OF_COLUMNS
    pattern_3_sum = boards_flat[:, ids] + boards_flat[:, ids + step] + boards_flat[:, ids + 2 * step] + \
                    boards_flat[:, ids + 3 * step]
    # pattern =    o
    #             o
    #            o
    #           o
    step = NUMBER_OF_COLUMNS - 1
    ids = np.tile(np.asarray([3, 4, 5, 6]), NUMBER_OF_ROWS - 3)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 4) * NUMBER_OF_COLUMNS
    pattern_4_sum = boards_flat[:, ids] + boards_flat[:, ids + step] + boards_flat[:, ids + 2 * step] + \
                    boards_flat[:, ids + 3 * step]

    winning_moves = np.sum(pattern_1_sum == 4*current_player, axis=1) + \
                    np.sum(pattern_2_sum == 4*current_player, axis=1) + \
                    np.sum(pattern_3_sum == 4*current_player, axis=1) + \
                    np.sum(pattern_4_sum == 4*current_player, axis=1)
    population_finished = np.zeros(boards.shape[0])
    population_finished[winning_moves > 0] = 1
    return population_finished


def population_evaluate_boards(boards, current_player):
    boards_flat = boards.copy().reshape((boards.shape[0], NUMBER_OF_COLUMNS*NUMBER_OF_ROWS))
    population_scores = np.zeros(POPULATION_SIZE, dtype=np.int32)

    # pattern = oooo
    step = 1
    ids = np.tile(np.asarray([0, 1, 2, 3]), NUMBER_OF_ROWS)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 4) * NUMBER_OF_COLUMNS
    pattern_1_sum = boards_flat[:, ids] + boards_flat[:, ids + step] + boards_flat[:, ids + 2 * step] +\
                    boards_flat[:, ids + 3 * step]
    # pattern = o
    #           o
    #           o
    #           o
    step = NUMBER_OF_COLUMNS
    ids = np.tile(np.asarray([0, 1, 2, 3, 4, 5, 6]), NUMBER_OF_ROWS - 3)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 7) * NUMBER_OF_COLUMNS
    pattern_2_sum = boards_flat[:, ids] + boards_flat[:, ids + step] + boards_flat[:, ids + 2 * step] +\
                    boards_flat[:, ids + 3 * step]
    # pattern = o
    #            o
    #             o
    #              o
    step = NUMBER_OF_COLUMNS + 1
    ids = np.tile(np.asarray([0, 1, 2, 3]), NUMBER_OF_ROWS - 3)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 4) * NUMBER_OF_COLUMNS
    pattern_3_sum = boards_flat[:, ids] + boards_flat[:, ids + step] + boards_flat[:, ids + 2 * step] +\
                    boards_flat[:, ids + 3 * step]
    # pattern =    o
    #             o
    #            o
    #           o
    step = NUMBER_OF_COLUMNS - 1
    ids = np.tile(np.asarray([3, 4, 5, 6]), NUMBER_OF_ROWS - 3)
    ids_temp = np.arange(ids.shape[0])
    ids = ids + (ids_temp // 4) * NUMBER_OF_COLUMNS
    pattern_4_sum = boards_flat[:, ids] + boards_flat[:, ids + step] + boards_flat[:, ids + 2 * step] +\
                    boards_flat[:, ids + 3 * step]

    for i in [-1, 1]:
        for j in [2, 3, 4]:
            id = i*j
            pattern_1_sum_temp = np.zeros_like(pattern_1_sum)
            pattern_1_sum_temp[pattern_1_sum == id] = 1
            pattern_2_sum_temp = np.zeros_like(pattern_2_sum)
            pattern_2_sum_temp[pattern_2_sum == id] = 1
            pattern_3_sum_temp = np.zeros_like(pattern_3_sum)
            pattern_3_sum_temp[pattern_3_sum == id] = 1
            pattern_4_sum_temp = np.zeros_like(pattern_4_sum)
            pattern_4_sum_temp[pattern_4_sum == id] = 1
            score = (np.sum(pattern_1_sum_temp, axis=1) + np.sum(pattern_2_sum_temp, axis=1) +
                                           np.sum(pattern_3_sum_temp, axis=1) + np.sum(pattern_4_sum_temp, axis=1))
            if j == 3:
                score *= 100
            elif j == 4:
                score *= 10000
            population_scores += i*score*current_player

    return population_scores


def breed_population(population, scores, possible_moves):
    # Sortowanie osobników względem przystosowania
    pop_ids = np.arange(population.shape[0])
    ids_with_scores = np.zeros((POPULATION_SIZE, 2))
    ids_with_scores[:, 0] = pop_ids
    ids_with_scores[:, 1] = scores
    ids_with_scores = np.flip(ids_with_scores[ids_with_scores[:, -1].argsort()], axis=0)
    sorted_ids = ids_with_scores[:, 0]
    # Wybór 50% najlepszych osobników
    # best_specimens = select_best_specimens_half(sorted_ids)
    # Wybór osobników za pomocą selekcji turniejowej
    best_specimens_ids = select_best_specimens_tournament(sorted_ids)
    best_specimens = population[best_specimens_ids]
    ids = np.arange(best_specimens.shape[0] // 2)
    best_specimens_1 = best_specimens[2 * ids]
    best_specimens_2 = best_specimens[2 * ids + 1]
    # Dobieranie osobników w pary i rozmnażanie ich
    pair_ids = np.arange(best_specimens.shape[0])
    np.random.shuffle(pair_ids)
    new_specimens_1 = np.zeros_like(best_specimens_1.flatten())
    new_specimens_2 = np.zeros_like(best_specimens_2.flatten())
    crossover_boards_array = np.random.rand(best_specimens_1.flatten().shape[0])
    crossover_boards_array[crossover_boards_array > (100 - CROSSOVER_CHANCE) / 100] = 1
    crossover_boards_array[crossover_boards_array < 1] = 0

    new_specimens_1[crossover_boards_array == 0] = best_specimens_1.flatten()[crossover_boards_array == 0]
    new_specimens_1[crossover_boards_array == 1] = best_specimens_2.flatten()[crossover_boards_array == 1]
    new_specimens_1 = new_specimens_1.reshape(best_specimens_1.shape)

    new_specimens_2[crossover_boards_array == 0] = best_specimens_2.flatten()[crossover_boards_array == 0]
    new_specimens_2[crossover_boards_array == 1] = best_specimens_1.flatten()[crossover_boards_array == 1]
    new_specimens_2 = new_specimens_2.reshape(best_specimens_2.shape)

    new_specimens = np.append(new_specimens_1, new_specimens_2, axis=0)

    # Mutowanie osobników
    mutated_specimens = mutate_new_specimens(new_specimens, possible_moves)
    new_population = np.asarray(np.append(best_specimens, mutated_specimens, axis=0), dtype=np.int8)
    return new_population


def select_best_specimens_half(population_sorted_ids):
    best_specimens_ids = population_sorted_ids[:population_sorted_ids.shape[0].shape[0] // 2]
    return best_specimens_ids


def select_best_specimens_tournament(population_sorted_ids):
    pair_ids = np.arange(population_sorted_ids.shape[0])
    np.random.shuffle(pair_ids)
    pair_ids_1 = pair_ids[np.arange(0, pair_ids.shape[0], 2)]
    pair_ids_2 = pair_ids[np.arange(1, pair_ids.shape[0], 2)]
    ids = np.arange(population_sorted_ids.shape[0] // 2)
    best_specimens_ids = np.zeros(population_sorted_ids.shape[0] // 2, dtype=np.int32)
    best_specimens_ids[ids] = np.minimum(pair_ids_1[ids], pair_ids_2[ids])
    return best_specimens_ids


def mutate_new_specimens(specimens, possible_moves):
    mutation_chance = np.random.randint(0, 100, specimens.shape)
    mutation_array = np.random.choice(possible_moves, specimens.shape)
    mutation_chance[mutation_chance > MUTATION_CHANCE] = 0
    mutation_chance[mutation_chance > 0] = 1
    mutated_specimens = -1 * specimens * (mutation_chance - 1) + mutation_array * mutation_chance
    return mutated_specimens


def print_board(board):
    board_str = ''
    for i in range(NUMBER_OF_ROWS):
        board_str += '\n++'
        for j in range(NUMBER_OF_COLUMNS):
            if i == 0:
                board_str += '=====+'
            else:
                board_str += '-----+'
        board_str += '+\n||'
        for j in range(NUMBER_OF_COLUMNS):
            if board[i, j] == 0:
                board_str += '     |'
            elif board[i, j] == 1:
                board_str += '11111|'
            elif board[i, j] == -1:
                board_str += '  2  |'
            else:
                board_str += '     |'
        board_str += '|\n||'
        for j in range(NUMBER_OF_COLUMNS):
            if board[i, j] == 0:
                board_str += '     |'
            elif board[i, j] == 1:
                board_str += '1   1|'
            elif board[i, j] == -1:
                board_str += '22222|'
            else:
                board_str += '     |'
        board_str += '|\n||'
        for j in range(NUMBER_OF_COLUMNS):
            if board[i, j] == 0:
                board_str += '     |'
            elif board[i, j] == 1:
                board_str += '11111|'
            elif board[i, j] == -1:
                board_str += '  2  |'
            else:
                board_str += '     |'
        board_str += '|'
    board_str += '\n++'
    for j in range(NUMBER_OF_COLUMNS):
        board_str += '=====+'
    board_str += '+\n |'
    for j in range(NUMBER_OF_COLUMNS):
        board_str += '  {}  |'.format(j + 1)  # A, B, C...
    print(board_str)
