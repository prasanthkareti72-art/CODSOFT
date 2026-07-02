import tkinter as tk
from tkinter import messagebox
import math

# ---------------------------
# Game Variables
# ---------------------------
board = [" " for _ in range(9)]
human = "X"
ai = "O"

human_score = 0
ai_score = 0
draw_score = 0

# ---------------------------
# Winner Check
# ---------------------------
def check_winner(player):
    wins = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]

    for combo in wins:
        if all(board[i] == player for i in combo):
            return True

    return False

# ---------------------------
# Draw Check
# ---------------------------
def is_draw():
    return " " not in board

# ---------------------------
# Minimax Algorithm
# ---------------------------
def minimax(is_maximizing):

    if check_winner(ai):
        return 1

    if check_winner(human):
        return -1

    if is_draw():
        return 0

    if is_maximizing:

        best_score = -math.inf

        for i in range(9):

            if board[i] == " ":
                board[i] = ai

                score = minimax(False)

                board[i] = " "

                best_score = max(score, best_score)

        return best_score

    else:

        best_score = math.inf

        for i in range(9):

            if board[i] == " ":
                board[i] = human

                score = minimax(True)

                board[i] = " "

                best_score = min(score, best_score)

        return best_score

# ---------------------------
# AI Move
# ---------------------------
def ai_move():

    best_score = -math.inf
    move = -1

    for i in range(9):

        if board[i] == " ":

            board[i] = ai

            score = minimax(False)

            board[i] = " "

            if score > best_score:
                best_score = score
                move = i

    board[move] = ai

    buttons[move].config(text=ai, state="disabled")

# ---------------------------
# Update Scoreboard
# ---------------------------
def update_score():
    score_label.config(
        text=f"Human: {human_score}   AI: {ai_score}   Draws: {draw_score}"
    )

# ---------------------------
# End Game
# ---------------------------
def finish_game(message):

    global human_score
    global ai_score
    global draw_score

    if message == "Human Wins!":
        human_score += 1

    elif message == "AI Wins!":
        ai_score += 1

    else:
        draw_score += 1

    update_score()

    messagebox.showinfo("Game Over", message)

    disable_buttons()

# ---------------------------
# Disable Board
# ---------------------------
def disable_buttons():

    for btn in buttons:
        btn.config(state="disabled")

# ---------------------------
# Human Move
# ---------------------------
def click(index):

    if board[index] != " ":
        return

    board[index] = human

    buttons[index].config(
        text=human,
        state="disabled"
    )

    if check_winner(human):
        finish_game("Human Wins!")
        return

    if is_draw():
        finish_game("Draw!")
        return

    ai_move()

    if check_winner(ai):
        finish_game("AI Wins!")
        return

    if is_draw():
        finish_game("Draw!")
        return

# ---------------------------
# Restart Game
# ---------------------------
def restart_game():

    global board

    board = [" " for _ in range(9)]

    for btn in buttons:

        btn.config(
            text="",
            state="normal"
        )

# ---------------------------
# New Match
# ---------------------------
def reset_scores():

    global human_score
    global ai_score
    global draw_score

    human_score = 0
    ai_score = 0
    draw_score = 0

    update_score()

    restart_game()

# ---------------------------
# Window
# ---------------------------
root = tk.Tk()

root.title("AI Tic Tac Toe")
root.geometry("420x550")
root.resizable(False, False)

# ---------------------------
# Title
# ---------------------------
title = tk.Label(
    root,
    text="TIC TAC TOE AI",
    font=("Arial", 22, "bold")
)

title.pack(pady=10)

# ---------------------------
# Score Label
# ---------------------------
score_label = tk.Label(
    root,
    text="Human: 0   AI: 0   Draws: 0",
    font=("Arial", 12)
)

score_label.pack()

# ---------------------------
# Board Frame
# ---------------------------
frame = tk.Frame(root)

frame.pack(pady=20)

buttons = []

# ---------------------------
# Create Buttons
# ---------------------------
for row in range(3):

    for col in range(3):

        index = row * 3 + col

        btn = tk.Button(
            frame,
            text="",
            width=6,
            height=3,
            font=("Arial", 24, "bold"),
            command=lambda i=index: click(i)
        )

        btn.grid(
            row=row,
            column=col,
            padx=5,
            pady=5
        )

        buttons.append(btn)

# ---------------------------
# Control Buttons
# ---------------------------
restart_btn = tk.Button(
    root,
    text="Restart Game",
    font=("Arial", 12),
    width=20,
    command=restart_game
)

restart_btn.pack(pady=10)

reset_btn = tk.Button(
    root,
    text="Reset Scores",
    font=("Arial", 12),
    width=20,
    command=reset_scores
)

reset_btn.pack(pady=5)

# ---------------------------
# Footer
# ---------------------------
footer = tk.Label(
    root,
    text="Unbeatable AI using Minimax Algorithm",
    font=("Arial", 10)
)

footer.pack(pady=15)

# ---------------------------
# Run
# ---------------------------
root.mainloop()
