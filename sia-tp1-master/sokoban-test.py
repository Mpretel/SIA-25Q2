import time
from collections import deque
import copy
import os

MOVES = {
    "w": (-1, 0),
    "s": (1, 0),
    "a": (0, -1),
    "d": (0, 1)
}

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def board_to_str(board):
    return "\n".join("".join(row) for row in board)

def find_player(board):
    for r, row in enumerate(board):
        for c, val in enumerate(row):
            if val in ("@", "+"):
                return r, c
    return None

def move(board, dr, dc):
    board = copy.deepcopy(board)
    r, c = find_player(board)
    nr, nc = r + dr, c + dc
    dest = board[nr][nc]

    if dest == "#":
        return None

    if dest in ("$", "*"):
        br, bc = nr + dr, nc + dc
        behind = board[br][bc]
        if behind in ("#", "$", "*"):
            return None
        board[br][bc] = "*" if behind == "." else "$"
        board[nr][nc] = "@" if dest == "$" else "+"
    else:
        board[nr][nc] = "@" if dest == " " else "+"

    board[r][c] = " " if board[r][c] == "@" else "."
    return board

def is_solved(board):
    return all(cell != "$" for row in board for cell in row)

def bfs_solve(start_board):
    start_state = board_to_str(start_board)
    queue = deque([(start_board, "")])
    visited = {start_state}

    while queue:
        board, path = queue.popleft()

        if is_solved(board):
            return path

        for move_key, (dr, dc) in MOVES.items():
            new_board = move(board, dr, dc)
            if new_board is None:
                continue
            state_str = board_to_str(new_board)
            if state_str not in visited:
                visited.add(state_str)
                queue.append((new_board, path + move_key))
    return None

def print_board(board):
    for row in board:
        print("".join(row))

def animate_solution(board, solution, delay=0.3):
    current = copy.deepcopy(board)
    clear_screen()
    print_board(current)
    time.sleep(delay)

    for step in solution:
        dr, dc = MOVES[step]
        current = move(current, dr, dc)
        clear_screen()
        print_board(current)
        time.sleep(delay)

# Ejemplo de nivel
level = [
    list("########"),
    list("#     .#"),
    list("# .$  $#"),
    list("# $$@  #"),
    list("#  .   #"),
    list("#    . #"),
    list("########")
]

solution = bfs_solve(level)
if solution:
    print("Solución encontrada:", solution)
    input("Presiona Enter para ver la animación...")
    animate_solution(level, solution, delay=0.3)
else:
    print("No se encontró solución")
