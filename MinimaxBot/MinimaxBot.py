import numpy as np
from Player import Player
from Taktikl import *

class MinimaxBot(Player):

    RECURSION_LIMIT = 99
    
    def __init__(self, game: Taktikl, is_white_player=True):
        self.game = game
        self.is_white_player = is_white_player
    
    def get_score(self, field: Field):
        winner = self.game.find_winner(field)
        if winner == None:
            return 0
        if winner == self.game.current_player:
            return 10
        return -10

    def get_all_moves(self, field: Field, current_player: int):
        moves = []
        for row in range(field.size):
            for column in range(field.size):
                if field.cells[row][column] == current_player:
                    moves.append({'row': row, 'column': column, 'dir': Move.UP})
                    moves.append({'row': row, 'column': column, 'dir': Move.RIGHT})
                    moves.append({'row': row, 'column': column, 'dir': Move.DOWN})
                    moves.append({'row': row, 'column': column, 'dir': Move.LEFT})
        return moves

    def minimax(self, field: Field, current_player: int, level=0):
        winner = self.game.find_winner(field)
        if winner != None or level > MinimaxBot.RECURSION_LIMIT:
            return self.get_score(field), None

        scores = []
        moves = []

        next_player = -current_player

        all_moves = self.get_all_moves(field, current_player)

        for move in all_moves:
            row, column, dir = move['row'], move['column'], move['dir']
            pos = Position(row, column)
            try:
                possible_field = self.game.make_move(field, current_player, pos, dir)
            except Exception as e:
                continue
            scores.append(self.minimax(possible_field, next_player, level=level+1)[0])
            moves.append(move)

        if self.game.current_player == current_player:
            max_index = np.argmax(np.array(scores))
            return scores[max_index], moves[max_index]
        else:
            min_index = np.argmin(np.array(scores))
            return scores[min_index], moves[min_index]