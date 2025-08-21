import os
import copy
import time
from collections import deque
import heapq


class Sokoban:
    MOVES = {
        "w": (-1, 0),
        "a": (0, -1),
        "d": (0, 1),
        "s": (1, 0)
    }

    def __init__(self, board, level):
        self.start_board = copy.deepcopy(board)
        self.board = copy.deepcopy(board)
        self.level = level
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

    def solve(self, method):
        """Solves the Sokoban puzzle using the specified search method.
        It considers repeated states and deadlocks."""

        if self.is_solved(self.start_board): # checks if the start puzzle is solved
            return ""
    
        if method == "bfs":
            frontier = deque([(copy.deepcopy(self.start_board), "")])  # queue to pop states at the front and append new states at the end (FIFO)
            pop_func = frontier.popleft                                # pops the state at the front
        elif method == "dfs":
            frontier = [(copy.deepcopy(self.start_board), "")]         # stack to append and pop new states at the end (LIFO)
            pop_func = frontier.pop                                    # pops the state at the end
            
        visited = {self.board_to_str(self.start_board)} # set of visited states

        while frontier:
            board, path = pop_func() # pops the state at the front or end of the frontier
            
            self.nodes_expanded += 1

            for key, (dr, dc) in self.MOVES.items(): # iterates over all possible moves
                new_path = path + key
                new_board, deadlock = self.move(board, dr, dc)

                if self.is_solved(new_board): # checks if the puzzle is solved
                    return new_path

                if deadlock: # If the move results in a deadlock, skip this state
                    continue

                state_str = self.board_to_str(new_board)

                if state_str not in visited:  # Check for repeated states
                    visited.add(state_str)                # Add new state to visited
                    frontier.append((new_board, new_path)) # Add new state to frontier
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
    def load_level(level):
        base_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{base_path}/levels/level{level}.txt", "r", encoding="utf-8") as f:
            return [list(line.rstrip("\n")) for line in f]

    level = input("Choose a level: ")
    board = load_level(level)
    mode = input("Mode (play/solve): ").lower()
    game = Sokoban(board, level)

    # Play mode
    if mode == "play":
        # Stores the start time
        start_time = time.time()
        solution = game.play_manual()
    # Solve mode
    else:
        method = input("Search method (bfs/dfs): ").lower()
        # Stores the start time
        start_time = time.time()
        solution = game.solve(method)

    # Stores the end time
    end_time = time.time()
    if solution:
        print(f"Solution found for level {level}!")
        print(f"Solution: {' '.join(solution)}")
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

# imprimir tambien profundidad del arbol, pueod animar el arbol y preguntar si hay que poner opcion de max depth


# mapear las letras wsda a flechitas
'''arrow_keys = {
    'w': '↑',
    'a': '←',
    's': '↓',
    'd': '→'
}'''

# agregar depth y ancho del arbol y poner max depth 

# bajar los niveles más comunes

# grabar pantalla de la solucion en la terminal 