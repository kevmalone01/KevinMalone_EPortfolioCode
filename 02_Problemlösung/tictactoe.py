"""
Tic-Tac-Toe mit KI (Minimax-Algorithmus)
---------------------------------------

Dieses Programm ermöglicht es, gegen den Computer Tic-Tac-Toe zu spielen.
Die KI verwendet den Minimax-Algorithmus und spielt damit immer optimal.

Bedienung:
- Du spielst X, der Computer spielt O.
- Gib deinen Zug als Zeile und Spalte ein (z.B. 1 2).
- Das Spiel gibt nach jedem Zug das Spielfeld aus und meldet Sieg, Niederlage oder Unentschieden.

KI-Strategie:
- Die KI berechnet alle möglichen Spielverläufe und wählt immer den besten Zug.
- Sie kann nicht verlieren und nutzt jede Siegchance.

Ideal, um das Prinzip von Minimax und KI-Entscheidungen bei kleinen Spielen zu verstehen.
"""

import random

def print_board(board):
    print("\n")
    for row in board:
        print(" | ".join(row))
        print("-" * 9)


def check_winner(board, player):
    # Zeilen, Spalten und Diagonalen prüfen
    for i in range(3):
        if all([cell == player for cell in board[i]]):
            return True
        if all([board[j][i] == player for j in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]):
        return True
    if all([board[i][2 - i] == player for i in range(3)]):
        return True
    return False


def is_full(board):
    return all(cell != " " for row in board for cell in row)


def get_free_positions(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == " "]


def player_move(board):
    while True:
        try:
            move = input("Dein Zug (Zeile Spalte, z.B. 1 2): ")
            row, col = map(int, move.strip().split())
            row -= 1
            col -= 1
            if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == " ":
                board[row][col] = "X"
                break
            else:
                print("Ungültiger Zug. Versuche es erneut.")
        except Exception:
            print("Bitte gib zwei Zahlen zwischen 1 und 3 ein, getrennt durch ein Leerzeichen.")


def minimax(board, depth, is_maximizing):
    if check_winner(board, "O"):
        return 1
    if check_winner(board, "X"):
        return -1
    if is_full(board):
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for (i, j) in get_free_positions(board):
            board[i][j] = "O"
            score = minimax(board, depth + 1, False)
            board[i][j] = " "
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for (i, j) in get_free_positions(board):
            board[i][j] = "X"
            score = minimax(board, depth + 1, True)
            board[i][j] = " "
            best_score = min(score, best_score)
        return best_score


def computer_move(board):
    best_score = -float('inf')
    best_move = None
    for (i, j) in get_free_positions(board):
        board[i][j] = "O"
        score = minimax(board, 0, False)
        board[i][j] = " "
        if score > best_score:
            best_score = score
            best_move = (i, j)
    if best_move:
        i, j = best_move
        board[i][j] = "O"
        print(f"Computer setzt auf Feld: {i+1} {j+1}")


def main():
    print("Willkommen zu Tic-Tac-Toe! Du spielst X, der Computer spielt O.")
    board = [[" " for _ in range(3)] for _ in range(3)]
    print_board(board)

    while True:
        # Spielerzug
        player_move(board)
        print_board(board)
        if check_winner(board, "X"):
            print("Herzlichen Glückwunsch, du hast gewonnen!")
            break
        if is_full(board):
            print("Unentschieden!")
            break

        # Computerzug
        computer_move(board)
        print_board(board)
        if check_winner(board, "O"):
            print("Der Computer hat gewonnen!")
            break
        if is_full(board):
            print("Unentschieden!")
            break

if __name__ == "__main__":
    main()

"""
Minimax-Algorithmus und KI-Strategie in diesem Programm:

1. Minimax-Algorithmus
Die KI verwendet den sogenannten Minimax-Algorithmus.
Das ist ein Verfahren, mit dem der Computer alle möglichen Spielzüge durchrechnet und immer den für ihn besten Zug auswählt.

Funktionsweise:
- Die KI simuliert alle möglichen Züge (und die Antworten des Gegners) bis zum Spielende.
- Für jeden möglichen Endzustand berechnet sie einen Wert:
  +1: Der Computer (O) gewinnt.
  -1: Der Mensch (X) gewinnt.
   0: Unentschieden.
- Die KI wählt dann den Zug, der zu dem für sie besten Ergebnis führt (maximaler Wert).

2. Umsetzung im Code
- Die Funktion minimax(board, depth, is_maximizing) prüft rekursiv alle möglichen Spielverläufe:
  - Ist der Computer am Zug (is_maximizing=True), sucht er das Maximum.
  - Ist der Mensch am Zug (is_maximizing=False), sucht er das Minimum.
- Die Funktion computer_move(board) probiert alle freien Felder aus, ruft für jedes den Minimax-Algorithmus auf und wählt den Zug mit dem besten Ergebnis.

3. Ergebnis
- Der Computer spielt jetzt "perfekt": Er macht keine Fehler, kann nicht verlieren und nutzt jede Siegchance.
- Das Spiel bleibt für dich als Mensch eine Herausforderung!

Zusammengefasst:
Die KI schaut alle Möglichkeiten voraus und entscheidet sich immer für den besten Zug. Das ist mit Minimax für kleine Spiele wie Tic-Tac-Toe sehr gut machbar.
""" 