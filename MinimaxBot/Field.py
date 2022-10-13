import Position

class Field:
    def __init__(self, size=4):
        self.size = size
        # set up initial configuration of the field
        # 0 - empty cell, 1 - black piece, -1 - white piece
        self.cells = [[0 for i in range(size)] for i in range(size)]
        self.cells[0] = [(-1)**i for i in range(size)]
        self.cells[-1] = [(-1)**(i+1) for i in range(size)]

    def show(self):
        for i in range(self.size):
            for j in range(self.size):
                print(self.cells[i][j], end="\t")
            print()

    def is_empty_at(self, pos: Position):
        return self.cells[pos.row][pos.column] == 0