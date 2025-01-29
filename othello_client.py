import socket
import sys
from Othello.othello import Othello
class OthelloClient:
    def __init__(self, host="vm33.cs.lth.se", port=9035, idstring="your_unique_id"):
        self.host = host
        self.port = port
        self.idstring = idstring
        self.sock = None
        self.color = None  # Will be set to 'd' (dark) or 'w' (white)
        self.game = Othello(N=8)  # Use the existing Othello class

    def connect(self):
        """Establish connection to the Othello server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print("[Connected to server]")
        except Exception as e:
            print(f"Connection error: {e}")
            sys.exit(1)

    def send(self, message):
        """Send a message to the server."""
        self.sock.sendall((message + "\n").encode("utf-8"))

    def receive(self):
        """Receive a message from the server."""
        return self.sock.recv(1024).decode("utf-8").strip()

    def handle_game(self):
        """Main loop to interact with the Othello server."""
        self.connect()

        # Initial handshake
        print(self.receive())  # "Hi! I am your othello server."
        print(self.receive())  # "What is your name?"
        self.send(self.idstring)  # Send our ID
        print(self.receive())  # "Hello idstring! Your current win count is #"
        print(self.receive())  # "Your time limit is # secs"

        # Choose color (only in testing mode)
        print(self.receive())  # "choose colour, 'd' for dark, 'w' for white."
        self.color = input("Choose color (d for dark, w for white): ").strip().lower()
        self.send(self.color)  # Send color choice
        print(self.receive())  # "you are dark" or "you are white"

        # Game loop
        while True:
            response = self.receive()
            print(f"[Server]: {response}")

            if response.startswith("error"):
                print("Invalid move or unexpected error. Exiting.")
                break

            if "opponent’s move" in response:
                move = self.receive()
                print(f"Opponent moved: {move}")
                if move != "PASS":
                    self.apply_move_from_server(move)

            elif "your move" in response:
                move = self.get_best_move()
                if move:
                    self.send(move)
                else:
                    self.send("PASS")

            elif "The game is finished" in response:
                print(self.receive())  # "White: #"
                print(self.receive())  # "Dark: #"
                print(self.receive())  # "White won | Dark won | Draw"
                break

    def apply_move_from_server(self, move):
        """Convert move format (e.g., 'd3') to board coordinates and update game state."""
        col = ord(move[0]) - ord('a')  # Convert letter to index
        row = int(move[1]) - 1         # Convert number to index
        self.game.apply_move(row, col)

    def get_best_move(self):
        """Use minimax or another strategy to choose the best move."""
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return None  # No valid move, must pass

        best_move = valid_moves[0]  # Placeholder: Replace with actual minimax logic
        row, col = best_move
        return f"{chr(col + ord('a'))}{row + 1}"  # Convert back to server format

if __name__ == "__main__":
    client = OthelloClient(idstring="your_unique_id")
    client.handle_game()
