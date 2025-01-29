import random
import multiprocessing
import time

class Othello:
    def __init__(self, N=8, time_limit=20):
        """Initialize the NxN board with the starting position.
        Having it dynamic with N as a variable in case we want to abstract it further and since it makes
        manual debugging way easier. """
        self.N = N  # 8 is standard
        self.time_limit = time_limit
        self.board = [[0] * N for _ in range(N)]
        m_low = N // 2 - 1 if N % 2 == 0 else N // 2  # middle low
        m_high = N // 2 if N % 2 == 0 else N // 2 + 1  # middle high
        self.board[m_low][m_low], self.board[m_high][m_high] = -1, -1  # White discs
        self.board[m_low][m_high], self.board[m_high][m_low] = 1, 1  # dark discs
        self.current_player = 1  # dark starts
        self.directions = [(1, 0), (-1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1),
                           (1, 1)]  # vertically, horizontally, and diagonally
        self.number_of_prunings = 0
        self.number_of_evaluations = 0

    def print_board(self):
        """Display the board in a human-readable format."""
        print("  " + " ".join(str(i) for i in range(self.N)))
        for i, row in enumerate(self.board):
            print(i, " ".join(self.get_symbol(cell) for cell in row))

    @staticmethod
    def get_symbol(value):
        """Return correct symbol for cell on board."""
        return "○" if value == 1 else "●" if value == -1 else "."

    def inside_board(self, r, c):
        """Check if coordinate (r,c) is inside the board"""
        return 0 <= r < self.N and 0 <= c < self.N

    def is_valid_move(self, row, col):
        """Check if a move is valid for the current player.
            Cond 1) Can not be on a cell that is already taken
            Cond 2) Must flip a disc - To flip a disc there must be a disc of your
             own color between a straight line fully contain by the opponents
             discs and the disc you put down. The line can be vertically, horizontally,
            or diagonally. All such lines that fulfill the requirement will become flipped.

        """
        # Cond 1
        if self.board[row][col] != 0:
            return False

        # Cond 2
        for direction_r, direction_c in self.directions:
            if self.can_flip_in_direction(row, col, direction_r, direction_c):
                return True
        return False

    def can_flip_in_direction(self, row, col, direction_r, direction_c):
        """Check if discs can be flipped in a given direction."""
        r, c = row + direction_r, col + direction_c  # first step in the direction
        opponent = -self.current_player
        found_opponent = False

        # traverses as long as opponent discs still on a straight line in given direction
        while self.inside_board(r, c) and self.board[r][c] == opponent:
            found_opponent = True
            # move additional step in direction
            r += direction_r
            c += direction_c

        # if next cell is our disc then we can flip
        if found_opponent and self.inside_board(r, c) and self.board[r][c] == self.current_player:
            return True
        # else if an empty cell, out of bound, or no opponent disc found we can not
        return False

    def apply_move(self, row, col, switch_player=False):
        """Place a disc and flip opponent's discs."""
        if not self.is_valid_move(row, col):  # check if valid
            return False

        self.board[row][col] = self.current_player
        for direction_r, direction_c in self.directions:
            if self.can_flip_in_direction(row, col, direction_r, direction_c):
                self.flip_discs_in_direction(row, col, direction_r, direction_c)
        if switch_player:
            self.current_player *= -1  # Switch turn
        return True

    def flip_discs_in_direction(self, row, col, direction_r, direction_c):
        """Flip opponent's discs in the given direction."""
        r, c = row + direction_r, col + direction_c  # first cell in our direction
        opponent = -self.current_player

        # traverse along the direction and switch opponent's discs to current player's
        while self.board[r][c] == opponent:
            self.board[r][c] = self.current_player
            r += direction_r
            c += direction_c

    def get_valid_moves(self):
        """Return a list of valid moves for the current player."""
        # for all possible moves return all valid ones
        return [(r, c) for r in range(self.N) for c in range(self.N) if self.is_valid_move(r, c)]

    def is_game_over(self):
        """Check if the game is over which is decided when neither player can move.
        Implemented such that it quickly gives false if not over since used a lot in minimax"""
        # Check if current player can move, if it can it will return False as soon as it finds a move,
        # speeding this function up

        # Checks if current player can move
        for r in range(self.N):
            for c in range(self.N):
                if self.is_valid_move(r, c):
                    # returns False as soon as it finds a move to speed up
                    return False

        self.current_player *= -1  # switch player
        # Checks if current player can move
        for r in range(self.N):
            for c in range(self.N):
                if self.is_valid_move(r, c):
                    self.current_player *= -1  # switch back player
                    return False
        self.current_player *= -1  # if we never switched backed inside for loop the switch back
        # now in case we want to do some calculation
        return True

    def get_score(self):
        """Get score of board with only one loop."""
        dark_score, white_score, unclaimed = 0, 0, 0
        for r in range(self.N):
            for c in range(self.N):
                value = self.board[r][c]
                if value > 0:
                    dark_score += 1
                elif value < 0:
                    white_score += 1
                else:
                    unclaimed += 1
        return dark_score, white_score, unclaimed

    def evaluate_board_state(self):
        """Evaluate the board state. Positive values for dark, negative values for white."""
        dark_score, white_score, unclaimed = self.get_score()
        return dark_score - white_score  # Higher values mean advantage for dark

    def get_winner_and_score(self):
        """Determine winner and score based on disc count. Unclaimed discs at the end of the game
        is given to the winner according to the rules on wikipedia."""
        dark_score, white_score, unclaimed = self.get_score()
        if dark_score == white_score:
            return f"Tie at score {dark_score}-{white_score}"
        elif dark_score > white_score:
            dark_score += unclaimed
            return f"First player (dark) won at score dark {dark_score}-{white_score} white"
        else:
            white_score += unclaimed
            return f"Second player (white) won at score dark {dark_score}-{white_score} white"

    def play_human_vs_human(self):
        """Run a basic game loop for human vs. human play."""
        while not self.is_game_over():
            self.print_board()
            valid_moves = self.get_valid_moves()
            if not valid_moves:
                print(f"No availabl moves for {self.get_symbol(self.current_player)}. Passing turn over.")
                self.current_player *= -1
                continue

            print(f"{self.get_symbol(self.current_player)} to move. Valid moves are: {valid_moves}")
            row, col = map(int, input(f'Enter row and col (e.g. "{valid_moves[0][0]} {valid_moves[0][1]}"): ').split())
            if (row, col) in valid_moves:
                self.apply_move(row, col, switch_player=True)
            else:
                print("Invalid move. Try again.")
        self.print_board()
        print(self.get_winner_and_score())

    def minimax(self, depth, alpha, beta, maximizing_player):
        # self.print_board()
        """Minimax algorithm with alpha-beta pruning."""
        # If leaf-node then evaluate current board state
        if depth == 0 or self.is_game_over():
            self.number_of_evaluations += 1
            return self.evaluate_board_state()

        # Get all valid moves
        valid_moves = random.shuffle(self.get_valid_moves())
        # If no valid moves we want to treat it like a pass and continue one depth further and see if the other player still can play
        if not valid_moves:
            return self.minimax(depth - 1, alpha, beta, not maximizing_player)

        # if dark, 1, we want to maximize
        if maximizing_player:
            max_eval = float('-inf')  # worst possible
            for move in valid_moves:
                original_board = [row[:] for row in self.board]  # deep copy

                self.apply_move(*move)
                evaluation = self.minimax(depth - 1, alpha, beta, False)  # get evaluation of our move
                self.board = original_board  # Reset to original board
                max_eval = max(max_eval, evaluation)  # choose the highest evaluation
                alpha = max(alpha,
                            evaluation)  # again choose the highest evaluation since this is what the maximizing player will do
                if beta <= alpha:  # The maximizing player will choose the highest value of alpha, since that is already higher than beta
                    break  # it doesn't matter if there is an even higher alpha since alpha will never be reached
                    # if it's above beta since the minimizing player will choose beta and we can therefore prune the rest of the options
            return max_eval

        # else if white, -1, we want to minimize
        else:
            min_eval = float('inf')  # worst possible
            for move in valid_moves:
                original_board = [row[:] for row in self.board]
                self.apply_move(*move)
                evaluation = self.minimax(depth - 1, alpha, beta, True)
                self.board = original_board  # Reset to original board
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:  # opposite now, we know if any beta is less than alpha, the maximizing player will just choose alpha anyways
                    break
            return min_eval

    def best_move(self, depth, ai_color):
        """Determine the best move for the current player using Minimax with alpha-beta pruning.
        Player dark, 1, that will say positive, wants to maximize.
        Player white, -1, that will sau negative, wants to minimize.
        If ai_color == 1 it wants to maximize
        If ai_color == 1 it wants to maximize
        """
        wants_to_maximize = ai_color == 1
        valid_moves = self.get_valid_moves()
        best_score = float('-inf') if wants_to_maximize else float('inf')
        best_move = None

        # print(valid_moves)
        # go through all moves to decide which one is best
        # print(f'current pl {self.current_player}')
        start_time = time.time()

        for move in valid_moves:
            if time.time() - start_time >= self.time_limit:
                print('stopped because of time limit')
                if best_move:
                    return best_move
                else:
                    return valid_moves[0]
            # print(move)
            original_board = [row[:] for row in self.board]  # deep copy of original board
            self.apply_move(*move)
            move_score = self.minimax(depth - 1, float('-inf'), float('inf'),
                                      wants_to_maximize)  # next move is the opponent,therefore if ai_color==1 we want it to False, but if ai_color==-1 the opponent is 1 and want to maximize so then we want it to be true
            self.board = original_board  # Reset board to original state
            # print(move_score)

            # If ai wants to maximize
            if wants_to_maximize and move_score > best_score:
                best_score = move_score
                best_move = move

            elif (not wants_to_maximize) and move_score < best_score:
                best_score = move_score
                best_move = move

        # print('best score is')
        # print(best_score)
        # print('----')
        # print(best_move)
        return best_move

    def play_human_vs_ai(self, depth, ai_player, randomize_simulation=False, print_board=True):
        """Allow a human to play against the AI, with a selectable AI color."""
        while not self.is_game_over():
            if print_board:
                self.print_board()
            # If the current player's turn is the AI
            if self.current_player == ai_player:

                # Get best move
                valid_moves = self.get_valid_moves()
                best_move = self.best_move(depth, ai_player)

                if best_move:
                    if self.apply_move(*best_move, switch_player=True):
                        if print_board:
                            print(f"Computer made move {best_move}.")
                    else:
                        print(f"WARNING: Said best move as not valid {best_move}")

                if valid_moves and not best_move:
                    print(f"WARNING: No best move but valid move exist")
                elif not valid_moves: # Only enters here if no move is available
                    self.current_player *= -1
                    if print_board:
                        print("No valid moves available. Passing on turn to human.")
                    continue

            # If the current player's turn is the human
            else:
                valid_moves = self.get_valid_moves()
                if not valid_moves:
                    if print_board:
                        print("No valid moves available. Passing on turn to ai.")
                    self.current_player *= -1
                    continue
                if print_board:
                    print(f"Your turn ({self.get_symbol(self.current_player)}). Valid moves: {valid_moves}")
                if not randomize_simulation:
                    row, col = map(int, input("Enter row and col (e.g., 3 2): ").split())
                else:  # in case we just want to simulate
                    row, col = random.choice(valid_moves)
                if (row, col) in valid_moves:
                    self.apply_move(row, col, switch_player=True)
                else:
                    print("Invalid move. Try again.")
        if print_board:
            self.print_board()
            print(self.get_winner_and_score())
        return self.get_winner_and_score()


def run_simulation(seed):
    print(f'Running simulation {seed}')
    game = Othello(time_limit=2)
    # random.seed(seed)
    ai_player = -1
    game.play_human_vs_ai(ai_player=ai_player, depth=20, randomize_simulation=True, print_board=False)
    print(game.get_winner_and_score())
    print(game.evaluate_board_state())
    return game.evaluate_board_state()


def run_parallel_simulations(K):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_simulation, range(K))

    dark, white, tied = 0, 0, 0
    for score in results:
        print(score)
        if score > 0:
            dark += 1
        elif score < 0:
            white += 1
        else:
            tied += 1
    print(f"Dark won: {dark} times, {round(100 * dark / K, 2)}%")
    print(f"White won: {white} times, {round(100 * white / K, 2)}%")
    print(f"Tied: {tied} times, {round(100 * tied / K, 2)}%")
    print(f"Number of matches: {K}")


if __name__ == "__main__":
    run_parallel_simulations(5)

    # ai_color = input("Choose AI color: '1' for dark, '-1' for white: ")
    # run_K_simulation(100)
    # game.play_human_vs_human()
