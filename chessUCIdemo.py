import chess
import numpy
import time

def get_prev_move(currentFEN, prevFEN):
	previous_board = chess.Board(prevFEN)
	current_board = chess.Board(currentFEN)
	final_move = None
	previous_board.castling_rights = chess.BB_A1 | chess.BB_H1 | chess.BB_A8 | chess.BB_H8

	##################################
	moves = previous_board.legal_moves
	if current_board == previous_board:
		pass
	else:
		for move in moves:
			Nf3 = chess.Move.from_uci(str(move))
			previous_board.push(Nf3)
			if str(previous_board.board_fen()) == str(current_board.board_fen()):
				final_move = Nf3
				break
			else:
				previous_board.pop()

	if final_move is not None: 
		return final_move

if __name__ == '__main__':
	
	bord = chess.Board('r1bqkbnr/ppppp1pp/2n5/5p2/3P4/5N2/PPP1PPPP/RNBQKB1R w - - 0 1')
	print(bord)
