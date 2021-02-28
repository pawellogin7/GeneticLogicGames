# -*- coding: utf-8 -*-

from os import system, name
import numpy as np
import matplotlib.pyplot as plt

# Stałe
TILE_COLORS = 6
TILES_IN_ROW = 4
POPULATION_SIZE = 10000
CROSSOVER_CHANCE = 50
MUTATION_CHANCE = 20
MAX_GAME_STEPS = 15
NUMBER_OF_GAMES = 1


def cls():
    # Windows
    if name == 'nt':
        _ = system('cls')
    # Mac and Linux
    else:
        _ = system('clear')


def start_game(mode, parameters, plot_games):
    global TILE_COLORS, TILES_IN_ROW, POPULATION_SIZE, CROSSOVER_CHANCE, MUTATION_CHANCE, MAX_GAME_STEPS, NUMBER_OF_GAMES
    TILE_COLORS, TILES_IN_ROW, POPULATION_SIZE, CROSSOVER_CHANCE, MUTATION_CHANCE, MAX_GAME_STEPS, NUMBER_OF_GAMES \
        = parameters
    if mode == 0:
        play_game()
    else:
        run_genetic_algorithm(plot_last_game=plot_games)


# Funkcje mastermind
def play_game():
    winning_row = random_winning_row()
    tiles_rows = []
    hint_rows = []
    game_step = 1

    while True:
        cls()
        print_game(tiles_rows, hint_rows)

        hints_array = np.asarray(hint_rows, dtype=np.int8)
        if hints_array.size != 0 and np.atleast_2d(hints_array)[hints_array.shape[0] - 1, 0] == TILES_IN_ROW:
            print('\nYou won! You have found secret combination in {}/{} guesses.\n'.format(game_step,
                                                                                            MAX_GAME_STEPS))
            print('Restart game by typing: rst')
            print('Quit game by typing: exit')
            print('\nWaiting for input')
            str_in = input()
            str_in = str_in.strip().replace(' ', '').upper()
            if str_in == 'EXIT':
                break
            elif str_in == 'RST':
                winning_row = random_winning_row()
                tiles_rows = []
                hint_rows = []
                game_step = 1
        elif game_step == MAX_GAME_STEPS:
            print('\nYou lost! You have not found secret combination in {} guesses.\n'.format(MAX_GAME_STEPS))
            print('Restart game by typing: rst')
            print('Quit game by typing: exit')
            print('\nWaiting for input')
            str_in = input()
            str_in = str_in.strip().replace(' ', '').upper()
            if str_in == 'EXIT':
                break
            elif str_in == 'RST':
                winning_row = random_winning_row()
                tiles_rows = []
                hint_rows = []
                game_step = 1
        else:
            print('\nCurrent guess: {}/{}\n'.format(game_step, MAX_GAME_STEPS))
            print('Make guess by typing combination of {} numbers from range <0;{}>'.format(TILES_IN_ROW, TILE_COLORS - 1))
            print('Restart game by typing: rst')
            print('Quit game by typing: exit')
            print('\nWaiting for input')
            str_in = input()
            str_in = str_in.strip().replace(' ', '').upper()
            input_row = np.zeros(TILES_IN_ROW, dtype=np.int8)
            if str_in == 'EXIT':
                break
            elif str_in == 'RST':
                winning_row = random_winning_row()
                tiles_rows = []
                hint_rows = []
                game_step = 1
            else:
                row_correct = True
                if len(str_in) != TILES_IN_ROW:
                    row_correct = False
                else:
                    for i in range(TILES_IN_ROW):
                        try:
                            tile = int(str_in[i])
                            if 0 <= tile < TILE_COLORS:
                                input_row[i] = tile
                            else:
                                row_correct = False
                                break
                        except (ValueError, IndexError):
                            row_correct = False
                            break
                if row_correct is False:
                    continue
                game_step += 1
                new_hints_row = process_row(input_row, winning_row)
                tiles_rows.append(input_row)
                hint_rows.append(new_hints_row)


def random_winning_row():
    winning_row = np.random.randint(0, TILE_COLORS, TILES_IN_ROW)
    return winning_row


def process_row(input_row, winning_row):
    perfect_match = np.sum(input_row == winning_row)
    color_match = 0
    color_match_winnig_row = winning_row.copy()
    for i in range(len(input_row)):
        try:
            match = np.where(color_match_winnig_row == input_row[i])[0][0]
            color_match_winnig_row[match] = -1 
            color_match += 1
        except (ValueError, IndexError):
            pass
    color_match -= perfect_match
    return [int(perfect_match), int(color_match)]


# Funkcje algorytmu genetycznego
def random_population():
    population = np.random.randint(0, TILE_COLORS, (POPULATION_SIZE, TILES_IN_ROW))
    return population


def evaluate_population(population, guesses, hints):
    population_with_scores = np.zeros((population.shape[0], population.shape[1] + 1))
    population_with_scores[:, :-1] = population
    population_score = check_population_match_score(population, guesses, hints)
    population_with_scores[:, population_with_scores.shape[1] - 1] = population_score
    population_with_scores = np.flip(population_with_scores[population_with_scores[:,
                                                            population_with_scores.shape[1] - 1].argsort()], axis=0)
    population = population_with_scores[:, :-1]
    scores = population_with_scores[:, -1]
    return population, scores


def check_population_match_score(population, guesses, hints):
    score = np.zeros(population.shape[0])
    ids = np.arange(population.shape[0])
    for guess, hint in zip(guesses, hints):
        perfect_match = np.zeros(population.shape[0])
        perfect_match[ids] = np.sum(population[ids] == guess, axis=1)
        population_colors = np.zeros((population.shape[0], TILE_COLORS))
        guess_colors = np.zeros(TILE_COLORS)
        for color_id in range(TILE_COLORS):
            population_colors[ids, color_id] = np.sum(population[ids] == color_id, axis=1)
            guess_colors[color_id] = np.sum(np.asarray(guess) == color_id)
        color_match = TILES_IN_ROW - np.sum(np.abs(population_colors - guess_colors) / 2, axis=1) - perfect_match
        specimen_score = 5 * np.abs(perfect_match - hint[0]) + np.abs(color_match - hint[1])
        score -= specimen_score
    return score


def breed_population(population, scores):
    # Wybór 50% najlepszych osobników
    # best_specimens = select_best_specimens_half(population)
    # Wybór osobników za pomocą selekcji turniejowej
    best_specimens = select_best_specimens_tournament(population)
    ids = np.arange(best_specimens.shape[0] // 2)
    best_specimens_first = best_specimens[2 * ids]
    best_specimens_second = best_specimens[2 * ids + 1]
    # Dobieranie osobników w pary i rozmnażanie ich
    pair_ids = np.arange(best_specimens.shape[0])
    np.random.shuffle(pair_ids)
    new_specimens_first = np.zeros_like(best_specimens_first.flatten())
    new_specimens_second = np.zeros_like(best_specimens_second.flatten())
    crossover_array = np.random.rand(best_specimens_first.flatten().shape[0])
    crossover_array[crossover_array > (100 - MUTATION_CHANCE) / 100] = 1
    crossover_array[crossover_array < 1] = 0

    new_specimens_first[crossover_array == 0] = best_specimens_first.flatten()[crossover_array == 0]
    new_specimens_first[crossover_array == 1] = best_specimens_second.flatten()[crossover_array == 1]
    new_specimens_first = new_specimens_first.reshape(best_specimens_first.shape)
    new_specimens_second[crossover_array == 0] = best_specimens_second.flatten()[crossover_array == 0]
    new_specimens_second[crossover_array == 1] = best_specimens_first.flatten()[crossover_array == 1]
    new_specimens_second = new_specimens_second.reshape(best_specimens_second.shape)
    new_specimens = np.append(new_specimens_first, new_specimens_second, axis=0)

    # Mutowanie osobników
    mutated_specimens = mutate_new_specimens(new_specimens)
    new_population = np.append(best_specimens, mutated_specimens, axis=0)
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
    best_specimens = np.zeros((population.shape[0] // 2, population.shape[1]))
    best_specimens[ids] = population[best_ids]
    return best_specimens


def mutate_new_specimens(specimens):
    mutation_chance = np.random.randint(0, 100, specimens.shape)
    mutation_array = np.random.randint(0, TILE_COLORS, specimens.shape)
    mutation_chance[mutation_chance > MUTATION_CHANCE] = 0
    mutation_chance[mutation_chance > 0] = 1
    mutated_specimens = -1 * specimens * (mutation_chance - 1) + mutation_array * mutation_chance
    return mutated_specimens


def run_genetic_algorithm(plot_last_game):
    games_scores = []
    current_game = 0
    while current_game < NUMBER_OF_GAMES:
        current_game += 1
        game_step = 1
        tiles_rows = []
        hint_rows = []

        winning_row = random_winning_row()
        mastermind_population = random_population()
        zero_score_specimens = []

        while game_step <= MAX_GAME_STEPS:
            cls()
            print_game(tiles_rows, hint_rows)
            print('\nGame {}/{}'.format(current_game, NUMBER_OF_GAMES))
            print('Genetic algorithm guess {}/{}\n'.format(game_step, MAX_GAME_STEPS))

            if len(tiles_rows) == 0:
                input_row = mastermind_population[0]
            else:
                mastermind_population, population_scores = evaluate_population(mastermind_population, tiles_rows,
                                                                               hint_rows)
                input_row = mastermind_population[0]
                zero_score_specimens.append(np.sum(population_scores == 0))
                mastermind_population = breed_population(mastermind_population, population_scores)
            new_hints_row = process_row(input_row, winning_row)
            tiles_rows.append(input_row)
            hint_rows.append(new_hints_row)

            if np.array_equal(input_row, winning_row) is True:
                cls()
                print_game(tiles_rows, hint_rows)
                print('\nGame {}/{}'.format(current_game, NUMBER_OF_GAMES))
                print('Genetic algorithm guess {}/{}\n'.format(game_step, MAX_GAME_STEPS))
                print('Genetic algorithm has found a solution in {} guesses.'.format(game_step))
                games_scores.append(game_step)
                break
            elif game_step >= MAX_GAME_STEPS:
                cls()
                print_game(tiles_rows, hint_rows)
                print('\nGame {}/{}'.format(current_game, NUMBER_OF_GAMES))
                print('Genetic algorithm guess {}/{}\n'.format(game_step, MAX_GAME_STEPS))
                print('Genetic algorithm was not able to find a solution in {} guesses.'.format(MAX_GAME_STEPS))
                games_scores.append(MAX_GAME_STEPS + 1)
                if NUMBER_OF_GAMES == 1:
                    print('Drawing new population...')
                    game_step = 1
                    tiles_rows = []
                    hint_rows = []
                    mastermind_population = random_population()
                    zero_score_specimens = []
            game_step += 1

        hint_rows = np.asarray(hint_rows)
        if current_game == NUMBER_OF_GAMES:
            if plot_last_game is True:
                guess_scores = -1 * (5 * TILES_IN_ROW - 5 * hint_rows[:, 0] - hint_rows[:, 1])
                zero_score_specimens = np.asarray(zero_score_specimens)

                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(14, 8)
                ax1.plot(np.arange(1, guess_scores.shape[0] + 1, 1), guess_scores)
                x_ticks = np.linspace(1, zero_score_specimens.shape[0] + 1, np.minimum(zero_score_specimens.shape[0] + 1
                                                                                       , 10), dtype=np.int32)
                x_ticks = np.append(x_ticks, zero_score_specimens.shape[0] + 1)
                ax1.set_xticks(x_ticks)
                ax1.set_title('Guess scores')
                ax1.set(xlabel='Guess', ylabel='Guess score')
                ax2.plot(np.arange(1, zero_score_specimens.shape[0] + 1, 1), zero_score_specimens)
                x_ticks = np.linspace(1, zero_score_specimens.shape[0], np.minimum(zero_score_specimens.shape[0], 10),
                                      dtype=np.int32)
                x_ticks = np.append(x_ticks, zero_score_specimens.shape[0])
                ax2.set_xticks(x_ticks)
                y_ticks = np.linspace(0, np.amax(zero_score_specimens), np.minimum(10, np.amax(zero_score_specimens)),
                                      dtype=np.int32)
                y_ticks = np.append(y_ticks, np.amax(zero_score_specimens))
                ax2.set_yticks(y_ticks)
                ax2.set_title('Potential solutions in population')
                ax2.set(xlabel='Population', ylabel='Potential solutions(specimens with score=0)')

                try:
                    plt.savefig('plots/mastermind_scores')
                except FileNotFoundError:
                    pass
                plt.show()

            while True:
                cls()
                print_game(tiles_rows, hint_rows)
                print('\nGame {}/{}'.format(current_game, NUMBER_OF_GAMES))
                print('Genetic algorithm guess {}/{}\n'.format(game_step, MAX_GAME_STEPS))
                if np.array_equal(tiles_rows[len(tiles_rows) - 1], winning_row) is True:
                    print('Genetic algorithm has found a solution in {} guesses.'.format(game_step))
                else:
                    print('Genetic algorithm was not able to find a solution in {} guesses.'.format(MAX_GAME_STEPS))

                if NUMBER_OF_GAMES > 1:
                    games_scores = np.asarray(games_scores)
                    print('\n\n==========================')
                    print("Genetic algorithm results for {} games".format(NUMBER_OF_GAMES))
                    print('--------------------------')
                    print('Number of solved games: {}/{}'.format(np.sum(games_scores != (MAX_GAME_STEPS + 1)),
                                                                 NUMBER_OF_GAMES))
                    print("Average guesses: {}".format(np.mean(games_scores)))
                    print("Least guesses: {}".format(np.amin(games_scores)))
                    print("Most guesses: {}".format(np.amax(games_scores[games_scores != MAX_GAME_STEPS + 1])))
                    print('==========================')

                print('\nRestart game by typing: rst')
                print('Quit game by typing: exit')
                print('\nWaiting for input')
                str_in = input()
                str_in = str_in.strip().replace(' ', '').upper()
                if str_in == 'EXIT':
                    break
                elif str_in == 'RST':
                    games_scores = []
                    current_game = 0
                    break


def print_game(guesses, hints):
    guesses = np.atleast_2d(np.asarray(guesses, dtype=np.int8))
    hints = np.atleast_2d(np.asarray(hints, dtype=np.int8))
    game_str = '|'
    for i in range((TILES_IN_ROW * 3 - 6) // 2):
        game_str += ' '
    game_str += 'Guess '
    for i in range((TILES_IN_ROW * 3 - 6) // 2):
        game_str += ' '
    game_str += '| Perfect |  Color  |\n|'
    for i in range(TILES_IN_ROW):
        game_str += '   '
    game_str += '|  Match  |  Match  |\n|'
    for i in range(TILES_IN_ROW):
        game_str += '==='
    game_str += '+=========+=========|\n'
    if guesses.size != 0:
        for i in range(guesses.shape[0]):
            game_str += '|'
            for j in range(TILES_IN_ROW):
                game_str += ' {} '.format(guesses[i, j])
            game_str += '|    {}    |    {}    |\n'.format(hints[i, 0], hints[i, 1])
        game_str += '|'
        for j in range(TILES_IN_ROW):
            game_str += '---'
        game_str += '+---------+---------|\n'
    print(game_str)

