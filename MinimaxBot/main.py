from Taktikl import *
from MinimaxBot import MinimaxBot

if __name__ == '__main__':
    t = Taktikl()
    bot = MinimaxBot(t)
    t.field.show()
    print('current_player', t.current_player)
    bot.minimax(t.field, t.current_player)