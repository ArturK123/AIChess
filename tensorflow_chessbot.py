import os
import chess
from chess_ui import UI
from chessUCIdemo import get_prev_move
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import mss
import mss.tools
import cv2
from helper_functions import unflipFEN
import helper_image_loading
import chessboard_finder
from stockfishpy.stockfishpy import *
from stockfish import Stockfish
import time
import json
import pyautogui
import argparse

prev_fen = None

def load_graph(frozen_graph_filepath):
    # Load and parse the protobuf file to retrieve the unserialized graph_def.
    with tf.io.gfile.GFile(frozen_graph_filepath, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import graph def and return.
    with tf.Graph().as_default() as graph:
        # Prefix every op/nodes in the graph.
        tf.import_graph_def(graph_def, name="tcb")
    return graph

def shortenFEN(fen):
  """Reduce FEN to shortest form (ex. '111p11Q' becomes '3p2Q')"""
  return fen.replace('11111111','8').replace('1111111','7') \
            .replace('111111','6').replace('11111','5') \
            .replace('1111','4').replace('111','3').replace('11','2')

class ChessboardPredictor(object):
  """ChessboardPredictor using saved model"""
  def __init__(self, frozen_graph_path='saved_models/frozen_graph.pb'):
    # Restore model using a frozen graph.
    graph = load_graph(frozen_graph_path)
    self.sess = tf.compat.v1.Session(graph=graph)

    # Connect input/output pipes to model.
    self.x = graph.get_tensor_by_name('tcb/Input:0')
    self.keep_prob = graph.get_tensor_by_name('tcb/KeepProb:0')
    self.prediction = graph.get_tensor_by_name('tcb/prediction:0')
    self.probabilities = graph.get_tensor_by_name('tcb/probabilities:0')

  def getPrediction(self, tiles):
    """Run trained neural network on tiles generated from image"""
    if tiles is None or len(tiles) == 0:
      print("Couldn't parse chessboard")
      return None, 0.0
    
    # Reshape into Nx1024 rows of input data, format used by neural network
    validation_set = np.swapaxes(np.reshape(tiles, [32*32, 64]),0,1)

    # Run neural network on data
    guess_prob, guessed = self.sess.run(
      [self.probabilities, self.prediction], 
      feed_dict={self.x: validation_set, self.keep_prob: 1.0})
    
    # Prediction bounds
    a = np.array(list(map(lambda x: x[0][x[1]], zip(guess_prob, guessed))))
    tile_certainties = a.reshape([8,8])[::-1,:]

    # Convert guess into FEN string
    # guessed is tiles A1-H8 rank-order, so to make a FEN we just need to flip the files from 1-8 to 8-1
    labelIndex2Name = lambda label_index: ' KQRBNPkqrbnp'[label_index]
    pieceNames = list(map(lambda k: '1' if k == 0 else labelIndex2Name(k), guessed)) # exchange ' ' for '1' for FEN
    fen = '/'.join([''.join(pieceNames[i*8:(i+1)*8]) for i in reversed(range(8))])
    return fen, tile_certainties

  def close(self):
    self.sess.close()

def get_fen_main(frontIMG, turn, predictor):
  tiles, corners = chessboard_finder.findGrayscaleTilesInImage(frontIMG)

  # Exit on failure to find chessboard in image
  if tiles is None:
    raise Exception('Couldn\'t find chessboard in image')
  
  fen, tile_certainties = predictor.getPrediction(tiles)
  short_fen = shortenFEN(fen)
  return ("%s %s - - 0 1" % (short_fen, str(turn)))

def get_moves(s):
  return s[:2], s[2:]

def click(from_, to):

    if from_[1] == '1':
        y = 1150
    elif from_[1] == '2':
        y = 1020
    elif from_[1] == '3':
        y = 890
    elif from_[1] == '4':
        y = 760
    elif from_[1] == '5':
        y = 625
    elif from_[1] == '6':
        y = 500
    elif from_[1] == '7':
        y = 375
    elif from_[1] == '8':
        y = 240
    else:
        x = None

    if from_[0] == 'a':
        x = 135
    elif from_[0] == 'b':
        x = 270
    elif from_[0] == 'c':
        x = 400
    elif from_[0] == 'd':
        x = 526
    elif from_[0] == 'e':
        x = 655
    elif from_[0] == 'f':
        x = 785
    elif from_[0] == 'g':
        x = 915
    elif from_[0] == 'h':
        x = 1048
    else:
        x = None

    if to[1] == '1':
        y2 = 1150
    elif to[1] == '2':
        y2 = 1020
    elif to[1] == '3':
        y2 = 890
    elif to[1] == '4':
        y2 = 760
    elif to[1] == '5':
        y2 = 625
    elif to[1] == '6':
        y2 = 500
    elif to[1] == '7':
        y2 = 375
    elif to[1] == '8':
        y2 = 240
    else:
        x2 = None

    if to[0] == 'a':
        x2 = 135
    elif to[0] == 'b':
        x2 = 270
    elif to[0] == 'c':
        x2 = 400
    elif to[0] == 'd':
        x2 = 526
    elif to[0] == 'e':
        x2 = 655
    elif to[0] == 'f':
        x2 = 785
    elif to[0] == 'g':
        x2 = 915
    elif to[0] == 'h':
        x2 = 1048
    else:
        x2 = None

    if len(to) > 2:
        if to[3] == 'q':
            pieceX = str(to[0])
            pieceY = str(to[1])
        elif to[3] == 'n':
            pieceX = str(to[0])
            pieceY = '375' 
        elif to[3] == 'r':
            pieceX = str(to[0])
            pieceY = '500'
        elif to[3] == 'b':
            pieceX = str(to[0])
            pieceY = '625'
    else:
        pieceX, pieceY = None, None

    return x, y, x2, y2, pieceX, pieceY

def mssIMG(screen):
    with mss.mss() as sct:
        moniter = {"top":screen[0], "left":screen[1], "width":screen[2], "height":screen[3]}
        output = "sct-{top}x{left}_{width}x{height}.png".format(**moniter)

        img = sct.grab(moniter)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

    return img

def boardAppend(board, move, moves):
    moves.append(move)
    go = chess.Move.from_uci(str(move))
    board.push(go)   

def updateTurnOpp(turn):
    if turn == 'w':
        turn_opp = 'b'
    elif turn == 'b':
        turn_opp = 'w' 

    return turn_opp

def myQuit(board, moves, predictor, happened=None):
    data = {}
    data['moves'] = []
    data['moves'].append(str(moves[0:]))
    data['board'] = []
    data['board'].append(str(board.board_fen()))
    with open('gameResults.json','w') as jsonFile:
        json.dump(data, jsonFile)
    print(happened)
    predictor.close()
    quit()

def main(args):
    #DO ONCE FOR BELOW
    predictor = ChessboardPredictor()
    screenGet = (172, 68, 1039, 1039)
    turn = 'w'
    player_turn = None
    board = chess.Board()
    chessEngine = Stockfish('/usr/local/Cellar/stockfish/12/bin/stockfish', parameters={'Threads':4, 'Ponder':'true'})
    moves = []
    move_if_black = None
    if args.c == 'w':
        pass
    elif args.c == 'b':
        move_if_black = str(args.move)
        turn = 'b'

    ###########################################
    if turn == 'w':

        img = mssIMG(screenGet)

        fen = get_fen_main(img, turn, predictor)

        chessEngine.set_fen_position(fen)
        move = chessEngine.get_best_move_time(1000)

        player_turn = 'b'

        before, after = get_moves(move)
        from_x, from_y, to_x, to_y, pieceX, pieceY = click(before, after)
        pyautogui.click(from_x, from_y)
        pyautogui.click(to_x, to_y)
        if pieceX is not None:
            pyautogui.click(pieceX, pieceY)

        boardAppend(board, move, moves)

        img2 = mssIMG(screenGet)

        turn_opp = updateTurnOpp(turn)

        prev_fen = str(get_fen_main(img2, turn_opp, predictor))

    else:
        boardAppend(board, move_if_black, moves)

        img = mssIMG(screenGet)

        fen = get_fen_main(img, turn, predictor)

        chessEngine.set_fen_position(fen)
        move = chessEngine.get_best_move_time(1000)

        before, after = get_moves(move)
        from_x, from_y, to_x, to_y, pieceX, pieceY = click(before, after)
        pyautogui.click(from_x, from_y)
        pyautogui.click(to_x, to_y)
        if pieceX is not None:
            pyautogui.click(pieceX, pieceY)

        boardAppend(board, move, moves)

        time.sleep(0.2)

        img2 = mssIMG(screenGet)

        turn_opp = updateTurnOpp(turn)

        prev_fen = str(get_fen_main(img2, turn_opp, predictor))

        player_turn = 'w'

    while True:
        if player_turn == turn:
            img = mssIMG(screenGet)

            fen = get_fen_main(img, turn, predictor)

            time.sleep(0.2)

            imgI = mssIMG(screenGet)

            fenI = get_fen_main(imgI, turn, predictor)

            if fen == fenI:
                pass
            else:
                fen = fenI

            chessEngine.set_fen_position(fen)
            move = chessEngine.get_best_move_time(1000)

            before, after = get_moves(move)
            from_x, from_y, to_x, to_y, pieceX, pieceY = click(before, after)
            pyautogui.click(from_x, from_y)
            pyautogui.click(to_x, to_y)
            if pieceX is not None:
                pyautogui.click(pieceX, pieceY)

            boardAppend(board, move, moves)

            img2 = mssIMG(screenGet)

            turn_opp = updateTurnOpp(turn)

            prev_fen = get_fen_main(img2, turn_opp, predictor)

            if player_turn == 'w':
                player_turn = 'b'
            elif player_turn == 'b':
                player_turn = 'w'

        elif player_turn != turn:

            img = mssIMG(screenGet)

            time.sleep(0.2)

            imgI = mssIMG(screenGet)

            cur_fen = get_fen_main(img, turn, predictor)

            cur_fenI = get_fen_main(imgI, turn, predictor)

            if cur_fen == cur_fenI:
                pass

            else:
                cur_fen = cur_fenI

            move_res = get_prev_move(cur_fenI, prev_fen)

            if move_res is not None:
                boardAppend(board, move_res, moves)
                if player_turn == 'w':
                    player_turn = 'b'
                elif player_turn == 'b':
                    player_turn = 'w'

            else:
                print('Opponent is thinking...')
                

        print(board)

        if str(board.is_check()) == 'True' and str(board.is_game_over()) == 'False':
            print('CHECK')
        elif str(board.is_checkmate()) == 'True':
            myQuit(board, moves, predictor, happened='Mate')
        elif str(board.is_stalemate()) == 'True':
            myQuit(board,moves, predictor, happened='Stale')

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    parser = argparse.ArgumentParser(description='What Color')
    parser.add_argument('--c', default='w', help='What color are you')
    parser.add_argument('--move', default=False, help='What move happened')
    args = parser.parse_args()
    main(args)