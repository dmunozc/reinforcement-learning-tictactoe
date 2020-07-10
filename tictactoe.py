"""Demonstration of reinforcement learning tic-tac-toe.

It uses table Q-learning to learn, and an absolute encoding of the game.
After 500k episodes it achieves a moderate good game strategy.
After learning, it lets you play against it.
The maximum possible test score it can get is 10. After 500k episodes, it
achieves an average of 6.5 points.
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import enum
import pandas as pd


class game_state(enum.Enum):
    """Enum class for game state."""

    WIN = 1
    DRAW = 2
    NA = 3


class Turn(enum.Enum):
    """Enum class for turns."""

    X_TURN = 0
    O_TURN = 1


WIN_REWARD = 1
LOSS_REWARD = 0
DRAW_REWARD = 0.5


def test_score(q_table, eps_prob, number_of_tests=10):
    """Performs play runs to measure ability of agent."""
    game_state = inital_game_board()
    final_score = 0
    for _ in range(number_of_tests):
        turn = Turn(random.randint(0, 1))
        while check_state(game_state) == game_state.NA:
            game_code = game_state_encoding(game_state)
            if turn == Turn.X_TURN:
                action_to_take = random.randint(0, 8)
                while not valid_move(game_state, action_to_take):
                    action_to_take = random.randint(0, 8)
                game_state[int(action_to_take / 3)][action_to_take % 3] = "X"
                turn = Turn.O_TURN
            else:
                actions = q_table[game_code]
                action_to_take = next_action(actions, game_state, eps_prob)
                game_state[int(action_to_take / 3)][action_to_take % 3] = "O"
                turn = Turn.X_TURN
        if check_state(game_state) == game_state.WIN:
            winner = get_winner(game_state)
            if winner == "O":
                final_score += 1
        else:
            final_score += 0.5
        game_state = inital_game_board()
    return final_score


def inital_game_board():
    """Empty tic tac toe board."""
    return [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]


def game_state_encoding(game_state):
    """Return encoded game board into an int to access the q_table."""
    res = ""
    for i in range(8, -1, -1):
        if game_state[int(i / 3)][i % 3] == " ":
            res += "00"
        if game_state[int(i / 3)][i % 3] == "X":
            res += "01"
        if game_state[int(i / 3)][i % 3] == "O":
            res += "10"
    return int(res, 2)


#
def game_state_decoding(game_code):
    """Decodes an int into a game board. For debugging purposes."""
    game_code_binary = "{0:b}".format(game_code)[::-1]
    game = inital_game_board()
    for i in range(0, 9 * 2, 2):
        loc = int(i / 2)
        step = game_code_binary[i : i + 2][::-1]
        if step == "00":
            game[int(loc / 3)][loc % 3] = " "
        if step == "01":
            game[int(loc / 3)][loc % 3] = "X"
        if step == "10":
            game[int(loc / 3)][loc % 3] = "O"
    return game


def check_state(game_state):
    """Return whether there is a winner, a draw, or the game continues."""
    empty_count = 0
    for i in range(9):
        if game_state[int(i / 3)][i % 3] == " ":
            empty_count += 1
    res = get_winner(game_state)
    if len(res) == 0:
        if empty_count == 0:
            return game_state.DRAW
        return game_state.NA
    return game_state.WIN


def valid_move(game_state, loc):
    """Returns if a move in a board is valid."""
    if game_state[int(loc / 3)][loc % 3] != " ":
        return False
    return True


def get_winner(game_state):
    """Returns the winner if any. Else it returns an empty string."""
    for i in range(len(game_state)):
        row = game_state[i]
        column = [game_state[0][i], game_state[1][i], game_state[2][i]]
        if row[0] == row[1] and row[0] == row[2] and row[0] != " ":
            return row[0]
        if (
            column[0] == column[1]
            and column[0] == column[2]
            and column[0] != " "
        ):
            return column[0]
    left_diag = [game_state[0][0], game_state[1][1], game_state[2][2]]
    right_diag = [game_state[2][0], game_state[1][1], game_state[0][2]]
    if (
        left_diag[0] == left_diag[1]
        and left_diag[0] == left_diag[2]
        and left_diag[0] != " "
    ):
        return left_diag[0]
    if (
        right_diag[0] == right_diag[1]
        and right_diag[0] == right_diag[2]
        and right_diag[0] != " "
    ):
        return right_diag[0]
    return ""


def print_board(game_state):
    """Prints the tic tac toe board."""
    for i in range(len(game_state)):
        row = game_state[i]
        print(row[0] + "|" + row[1] + "|" + row[2])
        if i < 2:
            print("-----")


def next_action(actions, game_state, eps_prob):
    """Returns action to take based on probability."""
    action_to_take = -1
    if not (np.sum(actions) == 0) and random.random() < (1 - eps_prob):
        possibleActions = np.argsort(actions)
        i = len(possibleActions) - 1
        while not valid_move(game_state, possibleActions[i]):
            i -= 1
        action_to_take = possibleActions[i]
    else:
        action_to_take = random.randint(0, len(actions) - 1)
        while not valid_move(game_state, action_to_take):
            action_to_take = random.randint(0, len(actions) - 1)
    return action_to_take


def main(alpha=0.9, eps_prob=0.3, episodes=10000):
    game_state = inital_game_board()
    # 174763 are all possible tic tac toe board combinations.
    # Some are unused.
    q_table = [[0 for _ in range(9)] for _ in range(174763)]
    eps_decrease_rate = 0.9
    scores = []
    print("Starting AI training for", episodes, "episodes")

    for j in range(1, episodes):
        if j % 10000 == 0:
            print("Episode", j)
        turn = random.randint(0, 1)
        previous_code = game_state_encoding(game_state)
        previous_action = 0
        if j % int(episodes / 12) == 0:
            eps_prob *= eps_decrease_rate
        while check_state(game_state) == game_state.NA:
            game_code = game_state_encoding(game_state)
            # AI does not learn from X turns. Uses its q_table for optimum
            # play but prefers random plays.
            if turn == Turn.X_TURN:
                actions = q_table[game_code]
                action_to_take = next_action(actions, game_state, 0.25)
                game_state[int(action_to_take / 3)][action_to_take % 3] = "X"
                turn = Turn.O_TURN
            else:
                actions = q_table[game_code]
                action_to_take = next_action(actions, game_state, eps_prob)
                game_state[int(action_to_take / 3)][action_to_take % 3] = "O"
                q_table[previous_code][previous_action] = q_table[
                    previous_code
                ][previous_action] + alpha * (
                    q_table[game_code][action_to_take]
                    - q_table[previous_code][previous_action]
                )
                previous_code = game_code
                game_code = game_state_encoding(game_state)
                previous_action = action_to_take
                turn = Turn.X_TURN

        if check_state(game_state) == game_state.WIN:
            winner = get_winner(game_state)
            # Only learn for O's.
            if winner == "O":
                reward = WIN_REWARD
            else:
                reward = LOSS_REWARD
        else:
            reward = DRAW_REWARD
        # Update q table.
        q_table[previous_code][previous_action] = q_table[previous_code][
            previous_action
        ] + alpha * (
            q_table[game_code][action_to_take]
            + reward
            - q_table[previous_code][previous_action]
        )
        # Reinitialize game.
        game_state = inital_game_board()
        scores.append(test_score(q_table, eps_prob))

    pd.Series(scores).rolling(100).mean().plot()
    plt.show()
    print("Human play.")
    print(
        "\nHow to play.\nEnter the number from 0 to 8 to place your "
        "move\nGame locations"
    )
    print_board([["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]])
    print("starting game\nPlayer is X, AI is O")
    while True:
        while check_state(game_state) == game_state.NA:
            if turn == Turn.X_TURN:
                print("Player's turn")
                loc = int(input("Location (0 to 8): "))
                while loc < 0 or loc > 8 or not valid_move(game_state, loc):
                    print("Invalid Location\nHere is the board")
                    print_board(
                        [["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]]
                    )
                    loc = int(input("Location (0 to 8): "))
                game_state[int(loc / 3)][loc % 3] = "X"
                turn = Turn.O_TURN
            else:
                print("AI's turn")
                game_code = game_state_encoding(game_state)
                actions = q_table[game_code]
                action_to_take = next_action(actions, game_state, 0)
                game_state[int(action_to_take / 3)][action_to_take % 3] = "O"
                turn = Turn.X_TURN
            print_board(game_state)

        if check_state(game_state) == game_state.WIN:
            winner = get_winner(game_state)
            print(winner + "'s won")
        else:
            print("It was a draw")
        game_state = inital_game_board()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", dest="alpha", help="alpha hyperparameter", default=0.9,
    )
    parser.add_argument(
        "-e", dest="eps", help="epsilon hyperparameter", default=0.3,
    )
    parser.add_argument(
        "-ep", dest="episode", help="number of episodes", default=500000,
    )
    args = parser.parse_args()
    main(float(args.alpha), float(args.eps), int(args.episode))
