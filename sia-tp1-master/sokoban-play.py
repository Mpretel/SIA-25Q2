import time
from collections import deque
import copy
import os


class Sokoban:
    MOVES = {
        "w": (-1, 0),
        "s": (1, 0),
        "a": (0, -1),
        "d": (0, 1)
    }


    def __init__(self, board, level):
        self.board = copy.deepcopy(board)
        self.level = level
        self.deadlock = False
        self.moves_count = 0
        self.moves_seq = []


    def find_player(self):
        """Finds the player's position in the board."""
        for r, row in enumerate(self.board):
            for c, val in enumerate(row):
                if val in ("@", "+"):
                    return r, c
        return None


    def is_solved(self):
        """Returns True if there are no boxes ($) left to place on their target"""
        return all(cell != "$" for row in self.board for cell in row)
    

    def clear_screen(self):
        """Clears the console screen."""
        os.system("cls" if os.name == "nt" else "clear")


    def print_board(self):
        """Prints the current state of the board and the number of moves made so far."""
        self.clear_screen()
        for row in self.board:
            print("".join(row))
        print(f"\nMovimientos: {self.moves_count}\n")


    def restore_cell(self, r, c):
        """Restores the player's previous cell"""
        if self.board[r][c] == "@":
            self.board[r][c] = " "
        elif self.board[r][c] == "+":
            self.board[r][c] = "."


    def check_deadlock(self, r, c, dr, dc):
        """Checks if the box at (r, c) is locked in a corner made of walls (#)"""
        behind_behind = self.board[r + dr][c + dc]
        if dc == 0:  # vertical move
            adj = (self.board[r][c+1], self.board[r][c-1])
        else:        # horizontal move
            adj = (self.board[r+1][c], self.board[r-1][c])
        if behind_behind == "#" and "#" in adj:
            return True
        return False
    

    def move(self, dr, dc, key):
        """Moves the player and the box (if any) in the specified direction.
        dr and dc are the row and column deltas.
        """
        # Current position of the player
        curr_r, curr_c = self.find_player()
        # New position of the player after the move
        new_r, new_c = curr_r + dr, curr_c + dc
        # Destination cell
        dest = self.board[new_r][new_c]

        if dest == "#":  # If the player is blocked by a wall, it cannot move
            return

        if dest in ("$", "*"):  # If a box is at the destination cell, check what's behind the box
            behind_r, behind_c = new_r + dr, new_c + dc
            behind = self.board[behind_r][behind_c]

            if behind in ("#", "$", "*"):  # If the box is blocked by a wall or another box, it cannot be pushed
                return

            # Move the box
            if behind == ".":
                self.board[behind_r][behind_c] = "*"  # Move box to goal
            else:
                self.board[behind_r][behind_c] = "$"  # Move box to empty space

            # Check if the box is pushed to a deadlock position: in that case the game will never be finished
            if behind == " " and self.check_deadlock(behind_r, behind_c, dr, dc):
                self.deadlock = True

        # Move player to new position
        if dest in (" ", "$"):
            self.board[new_r][new_c] = "@"  # The player moved to an empty space or pushed a box
        else:
            self.board[new_r][new_c] = "+"  # The player moved to a goal

        # Restore the player's previous cell
        self.restore_cell(curr_r, curr_c)

        # Increase the move count and store the selected key
        self.moves_count += 1  
        self.moves_seq.append(key)


    def play(self):
        """Starts the game loop."""

        # Stores the start time
        start_time = time.time()

        while True:
            self.print_board()

            if self.is_solved():
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"You won! Level {self.level} completed.")
                print(f"Total time: {elapsed:.1f} seconds")
                print(f"Number of moves: {self.moves_count}")
                print(f"Moves sequence: {' '.join(self.moves_seq)}")
                break

            if self.deadlock:
                end_time = time.time()
                elapsed = end_time - start_time
                print("A box is stuck in a deadlock! You lost.")
                break

            # Wait for a key press
            key = input("Press a key to move (↑W ←A ↓S →D): ").lower()
            if key in self.MOVES:
                dr, dc = self.MOVES[key]
                self.move(dr, dc, key)


if __name__ == "__main__":
    # Load the level from a file
    def load_level(level):
        base_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{base_path}/levels/level{level}.txt", "r", encoding="utf-8") as f:
            return [list(line.rstrip("\n")) for line in f]

    level = input("Choose the level: ")
    # define the relative path to the level file
    board = load_level(level)
    game = Sokoban(board, level)
    game.play()