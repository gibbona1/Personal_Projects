import tkinter as tk
import random

class PlinkoGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Plinko Game")
        
        self.balance = 10
        self.bet_amount = tk.DoubleVar()
        self.bet_amount.set(0.1)
        self.result_label = tk.Label(root, text=f"Balance: ${self.balance:.2f}")
        self.result_label.pack()

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()
        
        self.create_pegs()
        
        bet_frame = tk.Frame(root)
        bet_frame.pack()
        
        bet_label = tk.Label(bet_frame, text="Bet amount:")
        bet_label.pack(side=tk.LEFT)
        
        self.bet_entry = tk.Entry(bet_frame, textvariable=self.bet_amount)
        self.bet_entry.pack(side=tk.LEFT)
        
        bet_button = tk.Button(bet_frame, text="Place Bet", command=self.place_bet)
        bet_button.pack(side=tk.LEFT)
        
        self.play_button = tk.Button(root, text="Play", command=self.play)
        self.play_button.pack()
        
        self.result_text = tk.StringVar()
        self.result_message = tk.Label(root, textvariable=self.result_text)
        self.result_message.pack()
        
        self.ball = None
        self.ball_x = 200

    def create_pegs(self):
        self.peg_positions = []
        for row in range(9):
            for col in range(row + 1):
                x = 40 + col * 40 + (row % 2) * 20
                y = 50 + row * 40
                self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="black")
                self.peg_positions.append((x, y))

    def place_bet(self):
        try:
            bet = self.bet_amount.get()
            if bet <= self.balance:
                self.balance -= bet
                self.result_label.config(text=f"Balance: ${self.balance:.2f}")
                self.play_button.config(state=tk.NORMAL)
            else:
                self.result_text.set("Insufficient balance!")
        except ValueError:
            self.result_text.set("Invalid bet amount!")

    def play(self):
        self.play_button.config(state=tk.DISABLED)
        if self.ball:
            self.canvas.delete(self.ball)
        ball_y = 0
        self.drop_ball(self.ball_x, ball_y)

    def drop_ball(self, x, y):
        if y < 360:
            if self.ball:
                self.canvas.delete(self.ball)
            self.ball = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="blue")
            y += 10
            self.root.after(50, lambda: self.drop_ball(x, y))
        else:
            self.calculate_score(x)

    def calculate_score(self, x):
        slot = (x - 40) // 40
        score_multiplier = [1, 2, 3, 4, 5, 4, 3, 2, 1][slot]
        score = self.bet_amount.get() * score_multiplier
        self.balance += score
        self.result_label.config(text=f"Balance: ${self.balance:.2f}")
        self.result_text.set(f"Score: ${score:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    game = PlinkoGame(root)
    root.mainloop()
