import copy
def helper(x, y, no):
    # check all x
    for i in range(9):
        if grid[i][y] == no:
            return False
    # check y line
    for i in range(9):
        if grid[x][i] == no:
            return False

    # check sq
    xrem = x % 3
    yrem = y % 3

    for r in range(3):
        for c in range(3):
            if grid[x - xrem + r][y - yrem + c] == no:
                return False
    return True

def runSS(gg):
    global grid
    grid = gg
    sudokusolve()

    return pop
def sudokusolve():
    global grid

    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                for n in range(1, 10):

                    if helper(r, c, n) == True:


                        grid[r][c] = n  # allocate
                        #          print(grid)
                        sudokusolve()  # recursion
                        grid[r][c] = 0  # If it reaches here, it failed ahead and needs to back track.
                        # It goes back to the loop and becomes n+1 if that works.
                        # We need this line since if no other n works, 0 is required for future

                return  # reaches here when n is exhausted i,e 10 an nos possible moves left

    # return "Sol found"
    global pop
    pop= copy.deepcopy(grid)
