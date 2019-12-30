import random
import matplotlib.pyplot as plt 
import numpy as np
import sys
import enum

class GameState(enum.Enum):
  WIN = 1
  DRAW = 2
  NA = 3

class Turn(enum.Enum):
  X_TURN = 0
  O_TURN = 1

WIN_REWARD = 1
LOSS_REWARD = 0
DRAW_REWARD = 0.5


def getTestScore(qMatrix, epsProbability, numberOfTests=10):
  gameState = initalGameBoard()
  finalScore = 0
  for _ in range(numberOfTests):
    turn = Turn(random.randint(0, 1))
    while checkState(gameState) == GameState.NA:
      gameCode = getGameStateEncoding(gameState)
      if turn == Turn.X_TURN:
        actionToTake = random.randint(0, 8)
        while not checkIfValidMove(gameState, actionToTake):
          actionToTake = random.randint(0, 8)
        gameState[int(actionToTake/3)][actionToTake%3] = "X"
        turn = Turn.O_TURN
      else:
        actions = qMatrix[gameCode]
        actionToTake = getNextAction(actions, gameState, epsProbability)
        gameState[int(actionToTake/3)][actionToTake%3] = "O"
        turn = Turn.X_TURN
    if checkState(gameState) == GameState.WIN:
      winner = getWinner(gameState)
      if winner == "O":
        finalScore += 1
    else:
      finalScore += 0.5
    gameState = initalGameBoard()
  return finalScore
def initalGameBoard():
  return [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]

#encode the game board into an int to access the qmatrix
def getGameStateEncoding(gameState):
  res= ""
  for i in range(8, -1, -1):
    if gameState[int(i/3)][i%3] == " ":
      res += "00"
    if gameState[int(i/3)][i%3] == "X":
      res += "01"
    if gameState[int(i/3)][i%3] == "O":
      res += "10"
  return int(res, 2)

#decodes an int into a game board. For debugging purposes
def getGameStateDecoding(gameCode):
  gameCodeBinary = "{0:b}".format(gameCode)[::-1]
  game = initalGameBoard()
  for i in range(0, 9*2, 2):
    loc = int(i/2)
    step = gameCodeBinary[i:i+2][::-1]
    if step == "00":
      game[int(loc/3)][loc%3] = " "
    if step== "01":
      game[int(loc/3)][loc%3] = "X"
    if step == "10":
      game[int(loc/3)][loc%3] = "O"
  return game

#return wehter there is a winner, a draw, or the game continues
def checkState(gameState):
  emptyCount = 0
  for i in range(9):
    if gameState[int(i/3)][i%3] == " ":
      emptyCount += 1
  res = getWinner(gameState)
  if len(res) == 0:
    if  emptyCount == 0:
      return GameState.DRAW
    return GameState.NA
  return GameState.WIN

def checkIfValidMove(gameState, loc):
  if gameState[int(loc/3)][loc%3] != " ":
    return False
  return True

#returns the winner if any. Else it returns an empty string
def getWinner(gameState):
  for i in range(len(gameState)):
    row = gameState[i]
    column = [gameState[0][i], gameState[1][i], gameState[2][i]]
    if row[0] == row[1] and row[0] == row[2] and row[0] != " ":
      return row[0]
    if column[0] == column[1] and column[0] == column[2] and column[0] != " ":
      return column[0]
  leftDiag = [gameState[0][0], gameState[1][1], gameState[2][2]]
  rightDiag = [gameState[2][0], gameState[1][1], gameState[0][2]]
  if leftDiag[0] == leftDiag[1] and leftDiag[0] == leftDiag[2] and leftDiag[0] != " ":
      return leftDiag[0]
  if rightDiag[0] == rightDiag[1] and rightDiag[0] == rightDiag[2] and rightDiag[0] != " ":
      return rightDiag[0]
  return ""

def printBoard(gameState):
  for i in range(len(gameState)):
    row = gameState[i]
    print(row[0] + "|" + row[1] + "|" + row[2])
    if i < 2: print("-----")


def getNextAction(actions, gameState, epsProbability):
  actionToTake = -1
  if not (np.sum(actions) == 0) and random.random() < (1 - epsProbability):
    possibleActions = np.argsort(actions)
    i = len(possibleActions) - 1
    while not checkIfValidMove(gameState, possibleActions[i]):
      i -= 1
    actionToTake = possibleActions[i]
  else:
    actionToTake = random.randint(0, len(actions) - 1)
    while not checkIfValidMove(gameState, actionToTake):
      actionToTake = random.randint(0, len(actions) - 1)
      
  return actionToTake
   
def main(alpha=0.9, epsProbability=0.3, episodes=10000):   
  gameState = initalGameBoard()
  qMatrix = [[0 for _ in range(9)] for _ in range(174763)]
  epsDecreaseRate = 0.9
  scores = []
  
  print("Starting AI training for",episodes,"episodes")
  
  for j in range(1, episodes):
    if j % 5000 == 0:
      print(".", end='')
    turn = random.randint(0, 1)
    previousCode = getGameStateEncoding(gameState)
    previousAction  = 0
    
    if j % int(episodes/12) == 0:
      epsProbability*=epsDecreaseRate
      print(".",end='')
    
    while checkState(gameState) == GameState.NA:
      gameCode = getGameStateEncoding(gameState)
      #AI does not learn from X turns. Uses its qmatrix for optimum play but prefers
      #random plays
      if turn == Turn.X_TURN:
        actions = qMatrix[gameCode]
        actionToTake = getNextAction(actions, gameState, 0.25)
        gameState[int(actionToTake/3)][actionToTake%3] = "X"
        turn = Turn.O_TURN
      else:
        actions = qMatrix[gameCode]
        actionToTake = getNextAction(actions, gameState, epsProbability)
        gameState[int(actionToTake/3)][actionToTake%3] = "O"
        qMatrix[previousCode][previousAction] = qMatrix[previousCode][previousAction] + \
                                                  alpha * (qMatrix[gameCode][actionToTake] - \
                                                           qMatrix[previousCode][previousAction])
        previousCode = gameCode
        gameCode = getGameStateEncoding(gameState)
        previousAction = actionToTake
        turn = Turn.X_TURN
        
    if checkState(gameState) == GameState.WIN:
      winner = getWinner(gameState)
      #only learn for O's
      if winner == "O":
        reward = WIN_REWARD
      else:
        reward = LOSS_REWARD
    else:
      #draw reward
      reward = DRAW_REWARD
    #update last state
    qMatrix[previousCode][previousAction] = qMatrix[previousCode][previousAction] + \
                                              alpha * (qMatrix[gameCode][actionToTake] + \
                                                       reward - qMatrix[previousCode][previousAction])
    #reinitialize game
    gameState = initalGameBoard()
    scores.append(getTestScore(qMatrix, epsProbability))
   
  plt.plot(scores)
  plt.show()
  print("\nHow to play.\nEnter the number from 0 to 8 to place your move\nGame locations")
  printBoard([["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]])
  print("starting game\nPlayer is X, AI is O")
  while True:
    while checkState(gameState) == GameState.NA:
      if turn == Turn.X_TURN:
        print("Player's turn")
        loc = int(input("Location (0 to 8): "))
        while loc < 0 or loc > 8 or not checkIfValidMove(gameState, loc):
          print("Invalid Location\nHere is the board")
          printBoard([["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]])
          loc = int(input("Location (0 to 8): "))
        gameState[int(loc/3)][loc%3] = "X"
        turn = Turn.O_TURN
      else:
        print("AI's turn")
        gameCode = getGameStateEncoding(gameState)
        actions = qMatrix[gameCode]
        actionToTake = getNextAction(actions, gameState, 0)
        gameState[int(actionToTake/3)][actionToTake%3] = "O"
        turn = Turn.X_TURN
      printBoard(gameState)
        
    if checkState(gameState) == GameState.WIN:
      winner = getWinner(gameState)
      print(winner + "'s won")
    else:
      print("It was a draw")
    gameState = initalGameBoard()
    
    
if __name__ == '__main__':
  if len(sys.argv) > 3:
    main(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))
  else:
    print("Usage", sys.argv[0], "[alpha] [epsProbability] [episodes]")