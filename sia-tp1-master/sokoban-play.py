import copy

# Nivel inicial
level = [
    list("#######"),
    list("#     #"),
    list("# .$@ #"),
    list("#     #"),
    list("#######")
]

# Movimientos
MOVES = {
    "w": (-1, 0),
    "s": (1, 0),
    "a": (0, -1),
    "d": (0, 1)
}

def find_player(board):
    for r, row in enumerate(board):
        for c, val in enumerate(row):
            if val in ("@", "+"):
                return r, c
    return None

def move(board, dr, dc):
    r, c = find_player(board)
    nr, nc = r + dr, c + dc  # nueva posición jugador

    # Celda de destino
    dest = board[nr][nc]

    # Si hay pared → no hacer nada
    if dest == "#":
        return board

    # Si hay caja o caja sobre objetivo
    if dest in ("$", "*"):
        # Posición detrás de la caja
        br, bc = nr + dr, nc + dc
        behind = board[br][bc]

        # Si detrás de la caja hay pared o caja → no se puede empujar
        if behind in ("#", "$", "*"):
            return board

        # Mover la caja
        if behind == ".":
            board[br][bc] = "*"
        else:
            board[br][bc] = "$"

        # Actualizar donde estaba la caja
        board[nr][nc] = "@" if dest == "$" else "+"

    else:
        # Mover jugador a celda vacía o objetivo
        board[nr][nc] = "@" if dest == " " else "+"

    # Restaurar celda anterior del jugador
    if board[r][c] == "@":
        board[r][c] = " "
    else:
        board[r][c] = "."

    return board

def print_board(board):
    for row in board:
        print("".join(row))
    print()

def is_solved(board):
    for row in board:
        for cell in row:
            if cell == "$":
                return False
    return True

# Juego principal
board = copy.deepcopy(level)

while True:
    print_board(board)
    if is_solved(board):
        print("¡Nivel resuelto!")
        break
    move_key = input("Movimiento (WASD): ").lower()
    if move_key in MOVES:
        dr, dc = MOVES[move_key]
        board = move(board, dr, dc)
