from tensorflow_chessbot import ChessboardPredictor
import chess
predictor = ChessboardPredictor()
fen, certainty, _ = predictor.makePrediction('DemoBlack.png')
print("Predicted FEN: %s" % fen)
print("Certainty: %.1f%%" % (certainty*100))

board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1')
print(board)