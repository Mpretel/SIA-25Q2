import os
import copy
import time
from collections import deque
import heapq
import itertools


class Sokoban:
    MOVES = {
        "w": (-1, 0),  # Up
        "a": (0, -1),  # Left
        "s": (1, 0),   # Down
        "d": (0, 1)    # Right
    }

    def __init__(self, board):
        self.start_board = copy.deepcopy(board)
        self.board = copy.deepcopy(board)
        self.moves_seq = []
        self.nodes_expanded = 0

    def find_player(self, board):
        """Finds the player's position in the board."""
        for r, row in enumerate(board):
            for c, val in enumerate(row):
                if val in ("@", "+"):
                    return r, c

    def board_to_str(self, board):
        return "\n".join("".join(row) for row in board)

    def is_solved(self, board):
        """Returns True if there are no boxes ($) left to place on their target"""
        return all(cell != "$" for row in board for cell in row)

    def clear_screen(self):
        """Clears the console screen."""
        os.system("cls" if os.name == "nt" else "clear")
        
    def print_board(self, board=None):
        """Prints the current state of the board."""
        self.clear_screen()
        board = board or self.board
        for row in board:
            print("".join(row))
        print(f"\nMovement number: {len(self.moves_seq)}\n")

    def check_deadlock(self, board, r, c, dr, dc):
        """Checks if the box at (r, c) is locked in a corner made of walls (#)"""
        behind_behind = board[r + dr][c + dc]
        if dc == 0:  # vertical
            adj = (board[r][c+1], board[r][c-1])
        else:  # horizontal
            adj = (board[r+1][c], board[r-1][c])
        return behind_behind == "#" and "#" in adj

    def move(self, board, dr, dc):
        """Moves the player and the box (if any) in the specified direction.
        dr and dc are the row and column deltas.
        """
        new_board = copy.deepcopy(board)

        # Current position of the player
        curr_r, curr_c = self.find_player(new_board)
        # New position of the player after the move
        new_r, new_c = curr_r + dr, curr_c + dc
        # Destination cell
        dest = new_board[new_r][new_c]

        deadlock = False

        if dest == "#": # If the player is blocked by a wall, it cannot move
            return new_board, deadlock

        if dest in ("$", "*"): # If a box is at the destination cell, check what's behind the box
            behind_r, behind_c = new_r + dr, new_c + dc
            behind = new_board[behind_r][behind_c]

            if behind in ("#", "$", "*"): # If the box is blocked by a wall or another box, it cannot be pushed
                return new_board, deadlock

            # Move the box
            if behind == ".":
                new_board[behind_r][behind_c] = "*"  # Move box to goal
            else:
                new_board[behind_r][behind_c] = "$"  # Move box to empty space

            # Check if the box is pushed to a deadlock position
            if behind == " " and self.check_deadlock(new_board, behind_r, behind_c, dr, dc):
                deadlock = True

        # Move player to new position
        if dest in (" ", "$"):
            new_board[new_r][new_c] = "@"  # The player moved to an empty space or pushed a box
        else:
            new_board[new_r][new_c] = "+"  # The player moved to a goal

        # Restores the player's previous cell
        if new_board[curr_r][curr_c] == "@":
            new_board[curr_r][curr_c] = " " # Player was on an empty space
        else:
            new_board[curr_r][curr_c] = "." # Player was on a goal

        return new_board, deadlock

    def heuristic_manhattan(self, board):
        """Calculates the Manhattan distance heuristic for the given board state.
        Calculates the minimum sum of the Manhattan distances of each box ($) to its goal (.)
        """
        locs_objs = []
        locs_boxes = []
        for r, row in enumerate(board):
            for c, cell in enumerate(row):
                if cell == "$":
                    locs_boxes.append((r, c))
                elif cell == ".":
                    locs_objs.append((r, c))

        # Calculate distance of the combination of boxes and goals that gives the minimum distance 
        min_total_distance = float('inf')
        permutations = itertools.permutations(locs_boxes, len(locs_objs))
        for perm_locs_boxes in permutations:
            total_distance = 0
            for loc_box, loc_obj in zip(perm_locs_boxes, locs_objs):
                total_distance += abs(loc_box[0] - loc_obj[0]) + abs(loc_box[1] - loc_obj[1])
            if total_distance < min_total_distance:
                min_total_distance = total_distance

        return min_total_distance

    def heuristic_misplaced(self, board):
        """Calculates the misplaced tiles heuristic for the given board state.
        """
        # Counts the number of boxes in the wrong position ($)
        misplaced = sum(1 for row in board for cell in row if cell == "$")
        return misplaced

    def heuristic(self, board, strategy):
        """Calculates the heuristic value for the given board state.
        """
        # manhattan heuristic is always >= misplaced heuristic
        if strategy == "manhattan":
            return self.heuristic_manhattan(board)
        elif strategy == "misplaced":
            return self.heuristic_misplaced(board)
        else:
            raise ValueError(f"Unknown heuristic strategy: {strategy}")

    def solve(self, method, heuristic=None):
        """Solves the Sokoban puzzle using the specified search method.
        It considers repeated states and deadlocks."""

        heuristic_time = 0
        if self.is_solved(self.start_board): # checks if the start puzzle is solved
            return ""
    
        if method == "bfs":
            frontier = deque([(copy.deepcopy(self.start_board), "")])  # queue to pop states at the front and append new states at the end (FIFO)
            pop_func = frontier.popleft                                # pops the state at the front
        elif method == "dfs":
            frontier = [(copy.deepcopy(self.start_board), "")]         # stack to append and pop new states at the end (LIFO)
            pop_func = frontier.pop                                    # pops the state at the end
        else:
            # Metodos informados
            start = time.time()
            h = self.heuristic(board=self.start_board, strategy=heuristic)
            end = time.time()
            heuristic_time += (end - start)
            g = 0
            f = g + h
            frontier = [(f, copy.deepcopy(self.start_board), "")] # priority queue to pop states with the lowest cost (f = g + h)
            heapq.heapify(frontier)  # transform list into a heap
            pop_func = heapq.heappop # pops the state with the lowest cost

        visited = {self.board_to_str(self.start_board)} # set of visited states

        while frontier:
            if method == "bfs" or method == "dfs":
                board, path = pop_func() # pops the state at the front or end of the frontier
            else:
                f, board, path = pop_func(frontier) # pops the state with the lowest cost

            self.nodes_expanded += 1

            for key, (dr, dc) in self.MOVES.items(): # iterates over all possible moves
                new_path = path + key
                new_board, deadlock = self.move(board, dr, dc)

                if self.is_solved(new_board): # checks if the puzzle is solved
                    print(f"Heuristic calculation time: {heuristic_time} seconds")
                    return new_path

                if deadlock: # If the move results in a deadlock, skip this state
                    continue

                state_str = self.board_to_str(new_board)

                if state_str not in visited:  # Check for repeated states
                    visited.add(state_str)                # Add new state to visited
                    # Print the board for visualization (optional)
                    # self.print_board(new_board)
                    # print(len(visited), "nodes visited")
                    # time.sleep(0.1)  # Small delay to visualize the search process
                    
                    if method == "bfs" or method == "dfs":
                        frontier.append((new_board, new_path)) # Add new state to frontier
                    else:
                        start = time.time()
                        f = self.heuristic(board=new_board, strategy=heuristic) # h
                        end = time.time()
                        heuristic_time += (end - start)
                        if method == "a_star":
                            f += len(new_path) # g
                        heapq.heappush(frontier, (f, new_board, new_path))
        return None


    def replay_solution(self, solution, delay=0.3):
        """Replays the solution to the Sokoban puzzle. It replicates the player's moves and prints each board state."""
        board = copy.deepcopy(self.start_board)
        self.moves_seq = []
        for key in solution:
            dr, dc = self.MOVES[key]
            board, _ = self.move(board, dr, dc)
            self.moves_seq.append(key)
            self.print_board(board)
            time.sleep(delay)

    def play_manual(self):
        """Plays the Sokoban puzzle in manual mode."""
        self.moves_seq = [] # Initialize the moves sequence
        while True:
            self.print_board()

            if self.is_solved(self.board): # Check if the puzzle is solved
                return self.moves_seq

            # Get user input for the next move
            key = input("Press a key to move (↑W ←A ↓S →D): ").lower()
            if key in self.MOVES:
                dr, dc = self.MOVES[key]
                self.board, deadlock = self.move(self.board, dr, dc) # Move the player
                self.moves_seq.append(key) # Append the move to the sequence

                if deadlock: # If the move results in a deadlock, the game is over
                    self.print_board()
                    return None

if __name__ == "__main__":
    def ask_choice(prompt, choices):
        """Asks the user for a choice until a valid one is given."""
        choice = None
        choices_str = "/".join(choices)
        while choice not in choices:
            choice = input(f"{prompt} ({choices_str}): ").strip().lower()
        return choice

    def list_levels():
        """Lists all available Sokoban levels."""
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "levels")
        levels = [f.replace(".txt", "") for f in os.listdir(base_path) if f.endswith(".txt")]
        levels.sort()
        return levels

    def load_level(level):
        """Loads a Sokoban level from a file."""
        base_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{base_path}/levels/{level}.txt", "r", encoding="utf-8") as f:
            return [list(line.rstrip("\n")) for line in f]

    map_arrow_keys = {
        'w': '↑',
        'a': '←',
        's': '↓',
        'd': '→'
    }

    # Choose level
    available_levels = list_levels()
    level = ask_choice("Choose a level", available_levels)

    board = load_level(level)
    game = Sokoban(board)

    # Choose mode
    mode = ask_choice("Choose a mode", ["play", "solve"])

    # Play mode
    if mode == "play":
        start_time = time.time() # Stores the start time
        solution = game.play_manual()
    # Solve mode
    else:
        # Choose search method
        method = ask_choice("Choose a search method", ["bfs", "dfs", "greedy", "a_star"])

        # Choose heuristic strategy if needed
        strategy = None
        if method in ["greedy", "a_star"]:
            strategy = ask_choice("Choose an heuristic strategy", ["manhattan", "misplaced"])

        start_time = time.time() # Stores the start time
        solution = game.solve(method, strategy)

    end_time = time.time() # Stores the end time

    # Print results
    if solution:
        print(f"Solution found for level {level}!")
        arrow_solution = [map_arrow_keys[key] for key in solution] # map solution to arrow keys
        print(f"Solution: {' '.join(arrow_solution)}")
        print(f"Number of moves: {len(solution)}")
        if mode == "solve":
            print(f"Number of expanded nodes: {game.nodes_expanded}")
        elapsed = end_time - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds")
        input("Press Enter to replay the solution...")
        game.replay_solution(solution)
    else:
        if mode == "play":
            print("Deadlock! You lost.")
        else:
            print(f"No solution found for level {level}.")