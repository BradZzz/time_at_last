import numpy as np

w, h = 10, 10
players = [{
    'pos' : (0,0),
    'moves' : ['r','d','r','r','d','d','l','d','l','d','r','d','l','d','l','u','u','u']
},
{
    'pos' : (9,9),
    'moves' : ['l','u','l','u','u','l','l','u','l','d','d','d','l','d','l','l','u','l']
}]

playersB = [{
    'pos' : (0,0),
    'moves' : ['d','d','r','r','r','u','r','d','d','d','d','l','d','r','r','r','u','u']
},
{
    'pos' : (9,9),
    'moves' : ['l','l','l','u','u','l','u','u','u','r','r','u','l','l','l','l','u','r']
}]

playersC = [{
    'pos' : (0,0),
    'moves' : ['r','r','r','r','r','d','d','d','d','d','l','d','l','u','u','r','r','r']
},
{
    'pos' : (9,9),
    'moves' : ['l','u','l','u','l','l','u','u','r','u','l','u','u','l','l','d','d','r']
}]

def parseMove(move):
    return {
        'l': (-1,0),
        'r': (1,0),
        'u': (0,-1),
        'd': (0,1),
    }[move]

def playGame(board, players):
    loser = []
    max = len(players[0]['moves'])
    rnd = 0
    for round in range(0, max - 1):
        rnd = round
        for player in range(0, len(players)):
            if (player not in loser):
                curr = players[player]
                board[curr['pos'][0]][curr['pos'][1]] = player + 1
                curr['pos'] = tuple(map(sum,zip(curr['pos'],parseMove(curr['moves'][round]))))
                if (board[curr['pos'][0]][curr['pos'][1]] != 0 or
                        (curr['pos'][0] < 0 or curr['pos'][0] >= w or curr['pos'][1] < 0 or curr['pos'][1] >= h)):
                    loser += [player]
        if (players[0]['pos'] == players[1]['pos']):
            loser += [0, 1]
        if (len(loser) >= len(players) -1):
            break
    print "\n\nFinished!"
    print "Rounds: ", rnd
    print(np.matrix(board))

    for player in range(0, len(players)):
        if player in loser and len(loser) != len(players):
            print 'Loser: ', player + 1
        elif len(loser) + 1 == len(players):
            print 'Winner: ', player + 1
        else:
            print 'Draw: ', player + 1

board = [[0 for x in range(w)] for y in range(h)]
playGame(board, players)
board = [[0 for x in range(w)] for y in range(h)]
playGame(board, playersB)
board = [[0 for x in range(w)] for y in range(h)]
playGame(board, playersC)






