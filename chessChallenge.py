#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:59:32 2017

@author: laurenmurphy
"""

import numpy as np
import random

chessboard = np.arange(16).reshape((4, 4)) #create chessboard

all_locations = []
for x in range(4):
    for y in range(4):
        all_locations.append([x, y])    
available_moves = [[-2, 1],[-1, 2],[-2, -1],[-1, -2],[1, -2],[2, -1],[2, 1],[1, 2]]

def isMoveValid(location, move):
    new_location = np.sum([location,move],axis=0)
    if new_location[0] >= 0 and new_location[0] <= 3 and new_location[1] >= 0 and new_location[1] <= 3:
        return True
    else:
        return False
    
possible_moves = {}

for x in range(4): 
    for y in range(4):
        possible_moves[(x, y)] = []
        for move in available_moves:
            if isMoveValid((x,y), move) == True:
                possible_moves[(x,y)].append(move)                
                
def moveKnight(knight_loc):
    x = random.randint(0, len(possible_moves[(knight_loc)])-1)
    move = possible_moves.get(knight_loc)
    knight_loc = np.sum([knight_loc,move[x]],axis=0)
    return knight_loc

def getBoardValue(knight_loc):
    knight_loc = moveKnight(knight_loc)
    loc = chessboard[knight_loc[0],knight_loc[1]]
    return loc

knight = (0, 0)
S = 0
T = 512
modulo = 311
mod = []
for x in range(T):
    S = getBoardValue(knight) + S
    mod += [S % modulo]
print(S)
print(np.mean(mod))
print(np.std(mod))
print('%.10f' % np.mean(mod))
print('%.10f' % np.std(mod))                