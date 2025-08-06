import chess

def get_game_phase_name(board: chess.Board) -> str:
    """判斷遊戲階段（開局、中局、殘局）"""
    # 透過計算主要和次要子力的數量來判斷遊戲階段
    majors_and_minors = sum(len(board.pieces(pt, c)) for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT] for c in [chess.WHITE, chess.BLACK])
    
    # 如果已經進入中局且子力少於等於6，則判定為殘局
    if hasattr(get_game_phase_name, "is_middlegame") and get_game_phase_name.is_middlegame and majors_and_minors <= 6:
        return 'endgame'
    
    # 計算雙方底線的棋子數量
    white_backrank = sum(1 for sq in chess.SQUARES[:8] if board.piece_at(sq))
    black_backrank = sum(1 for sq in chess.SQUARES[56:] if board.piece_at(sq))
    backrank_sparse = white_backrank < 4 or black_backrank < 4
    
    # 透過混合度判斷是否為開局
    mixedness = sum(len(board.pieces(pt, chess.WHITE)) * len(board.pieces(pt, chess.BLACK)) for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    
    # 如果子力大於10，底線不稀疏，且混合度低，則為開局
    if majors_and_minors > 10 and not backrank_sparse and mixedness <= 150:
        get_game_phase_name.is_middlegame = False
        return 'opening'
    else:
        # 否則，設定為中局，並在子力過少時轉為殘局
        get_game_phase_name.is_middlegame = True
        return 'endgame' if majors_and_minors <= 6 else 'middlegame'

def get_eval_from_info(info: dict) -> int:
    """從 Stockfish 的分析資訊中提取評估分數"""
    score = info.get("score").white()
    # 如果是將殺局面，回傳極大或極小值
    if score.is_mate():
        return 10000 if score.mate() > 0 else -10000
    return score.score()

def get_current_ply(board: chess.Board) -> int:
    """計算當前局面的 ply 數"""
    # Ply 的計算方式：(回合數 * 2) - (白方為2，黑方為1)
    return board.fullmove_number * 2 - (2 if board.turn == chess.WHITE else 1)

def get_piece_values_for_ply(ply_count: int, piece_theoretical_values: dict, color: str = 'white') -> dict:
    """根據 ply 數和顏色取得對應的棋子價值"""
    # 如果有精確的 ply 對應值，直接回傳
    if ply_count in piece_theoretical_values:
        return piece_theoretical_values[ply_count][color]
    
    # 否則，尋找最接近的 ply
    available_plys = list(piece_theoretical_values.keys())
    closest_ply = min(available_plys, key=lambda x: abs(x - ply_count))
    return piece_theoretical_values[closest_ply][color]

def get_piece_mobility_for_ply(ply_count: int, piece_theoretical_mobility: dict, color: str = 'white') -> dict:
    """根據 ply 數和顏色取得對應的棋子機動性"""
    # 如果有精確的 ply 對應值，直接回傳
    if ply_count in piece_theoretical_mobility:
        return piece_theoretical_mobility[ply_count][color]
    
    # 否則，尋找最接近的 ply
    available_plys = list(piece_theoretical_mobility.keys())
    closest_ply = min(available_plys, key=lambda x: abs(x - ply_count))
    return piece_theoretical_mobility[closest_ply][color]
