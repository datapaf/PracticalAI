from Field import Field
from Position import Position
from Move import Move

class Taktikl:
    def __init__(self):
        self.field = Field()
        self.current_player = -1

    def make_move(self, field: Field, current_player: int, piece_pos: Position, move: str):
        
        def report_incorrect_move():
            raise Exception("Cannot make such move")
        
        def put_to_new_position(new_pos: Position):
            field.cells[new_pos.row][new_pos.column] = field.cells[piece_pos.row][piece_pos.column]
            field.cells[piece_pos.row][piece_pos.column] = 0
            return field

        if field.cells[piece_pos.row][piece_pos.column] != current_player:
            report_incorrect_move()

        if move == Move.UP:
            new_pos = Position(piece_pos.row - 1, piece_pos.column)
            if not new_pos.row >= 0 or not field.is_empty_at(new_pos):
                report_incorrect_move()
            return put_to_new_position(new_pos)
        
        elif move == Move.RIGHT:
            new_pos = Position(piece_pos.row, piece_pos.column + 1)
            if not new_pos.column < field.size or not field.is_empty_at(new_pos):
                report_incorrect_move()
            return put_to_new_position(new_pos)

        elif move == Move.DOWN:
            new_pos = Position(piece_pos.row + 1, piece_pos.column)
            if not new_pos.row < field.size or not field.is_empty_at(new_pos):
                report_incorrect_move()
            return put_to_new_position(new_pos)
        
        elif move == Move.LEFT:
            new_pos = Position(piece_pos.row, piece_pos.column - 1)
            if not new_pos.column >= 0 or not field.is_empty_at(new_pos):
                report_incorrect_move()
            return put_to_new_position(new_pos)

        else:
            report_incorrect_move()

    def find_winner(self, field: Field):

        # 1 - black pieces in row, -1 - white pieces in row,
        # None - no pieces in row
        def check_three_in_row(values: list):
            for i in range(len(values)-2):
                if values[i] == 0:
                    continue
                if values[i] == values[i+1] and values[i+1] == values[i+2]:
                    return values[i]
            return None

        values_to_check = []

        # horizontal values
        for row in range(field.size):
            values = field.cells[row]
            values_to_check.append(values)

        # vertical values
        for column in range(field.size):
            values = [field.cells[i][column] for i in range(field.size)]
            values_to_check.append(values)

        # main diagonal values
        for row in range(field.size - 2):
            for column in range(field.size - 2):
                values = [field.cells[row+i][column+i] for i in range(3)]
                values_to_check.append(values)
        
        # secondary diagonal values
        for row in range(field.size - 2, field.size):
            for column in range(field.size - 2):
                values = [field.cells[row-i][column+i] for i in range(3)]
                values_to_check.append(values)

        for values in values_to_check:
            winner = check_three_in_row(values)
            if winner == None:
                continue
            else:
                return winner

        return None

    def get_player_name(self, player: int):
        if player == -1:
            return 'white'
        if player == 1:
            return 'black'
        raise Exception("Unknown player")

    def get_move_from_user(self):
        raw = input(f"Enter piece's row and column and the move: ")
        row, column, move = raw.split()
        row = int(row)
        column = int(column)
        return row, column, move

    def change_turn(self):
        self.current_player *= -1

    def run(self):
        while True:
            self.field.show()
            print(f"Now moves: {self.get_player_name(self.current_player)}")
            row, column, move = self.get_move_from_user()
            pos = Position(row, column)
            try:
                self.field = self.make_move(
                    self.field, self.current_player, pos, move
                )
            except Exception as e:
                print(e)
                continue

            winner = self.find_winner(self.field)
            if winner != None:
                self.field.show()
                print(f"Winner: {self.get_player_name(winner)}")
                break

            self.change_turn()

    def run_with_bot(self, bot: MinimaxBot):
        while True:
            self.field.show()
            print(f"Now moves: {self.get_player_name(self.current_player)}")
            if self.current_player == -1 and bot.is_white_player:
                bot_move = bot.minimax(self.field, self.current_player)[1]
                print(bot_move)
                row, column, move = bot_move['row'], bot_move['column'], bot_move['dir']
            else:
                row, column, move = self.get_move_from_user()
            pos = Position(row, column)
            self.field = self.make_move(self.field, self.current_player, pos, move)

            winner = self.find_winner()
            if winner != None:
                self.field.show()
                print(f"Winner: {self.get_player_name(self.current_player)}")
                break

            self.change_turn()