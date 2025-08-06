import chess
import chess.engine
import os
import time
import json
import pandas as pd
from typing import List, NamedTuple
from KMAPS import KMAPS
from get_llm_analysis import get_llm_analysis, ChessAnalysis
from utils import get_game_phase_name, get_eval_from_info, get_current_ply, get_piece_values_for_ply, get_piece_mobility_for_ply

# ==============================================================================
# 1. 全域變數設定
# ==============================================================================
STOCKFISH_PATH = r"D:\python\my chess game code\stockfish\stockfish-windows-x86-64-avx2.exe"
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash-lite"
DEPTH = 12

# ==============================================================================
# 2. 價值體系和機動性體系 - 從 CSV 文件載入
# ==============================================================================
def load_piece_values_from_csv(csv_file_path="piece_values_simple.csv"):
    """
    從 CSV 文件載入棋子價值數據，按 ply 和顏色索引
    CSV 格式: ply,piece_type,color,average_value
    如果沒有color欄位，則假設是舊格式，黑白共用
    """
    try:
        df = pd.read_csv(csv_file_path)
        
        # 檢查是否有color欄位
        has_color_column = 'color' in df.columns
        
        # 創建按 ply 和顏色索引的價值字典
        piece_values = {}
        
        for _, row in df.iterrows():
            ply = int(row['ply'])
            piece_type = row['piece_type'].lower()
            value = int(row['average_value'])
            
            if has_color_column:
                color = row['color'].lower()  # 'white' 或 'black'
                
                if ply not in piece_values:
                    piece_values[ply] = {'white': {}, 'black': {}}
                
                if color not in piece_values[ply]:
                    piece_values[ply][color] = {}
                    
                piece_values[ply][color][piece_type] = value
            else:
                # 舊格式，黑白共用
                if ply not in piece_values:
                    piece_values[ply] = {'white': {}, 'black': {}}
                
                piece_values[ply]['white'][piece_type] = value
                piece_values[ply]['black'][piece_type] = value
        
        if has_color_column:
            print(f"成功從 CSV 文件載入 {len(piece_values)} 個 ply 的分色棋子價值數據")
        else:
            print(f"成功從 CSV 文件載入 {len(piece_values)} 個 ply 的棋子價值數據（黑白共用）")
        return piece_values
        
    except FileNotFoundError:
        print(f"警告: 找不到 CSV 文件 {csv_file_path}，使用預設價值")
        return {6: {'white': {'p': 151, 'n': 549, 'b': 600, 'r': 541, 'q': 677},
                   'black': {'p': 151, 'n': 549, 'b': 600, 'r': 541, 'q': 677}}}
    except Exception as e:
        print(f"載入 CSV 文件時發生錯誤: {e}，使用預設價值")
        return {6: {'white': {'p': 151, 'n': 549, 'b': 600, 'r': 541, 'q': 677},
                   'black': {'p': 151, 'n': 549, 'b': 600, 'r': 541, 'q': 677}}}

def load_piece_mobility_from_csv(csv_file_path="piece_mobility_simple.csv"):
    """
    從 CSV 文件載入棋子機動性數據，按 ply 和顏色索引
    CSV 格式: ply,piece_type,color,average_mobility
    如果沒有color欄位，則假設是舊格式，黑白共用
    """
    try:
        df = pd.read_csv(csv_file_path)
        
        # 檢查是否有color欄位
        has_color_column = 'color' in df.columns
        
        # 創建按 ply 和顏色索引的機動性字典
        piece_mobility = {}
        
        for _, row in df.iterrows():
            ply = int(row['ply'])
            piece_type = row['piece_type'].lower()
            mobility = float(row['average_mobility'])  # 機動性可能是小數
            
            if has_color_column:
                color = row['color'].lower()  # 'white' 或 'black'
                
                if ply not in piece_mobility:
                    piece_mobility[ply] = {'white': {}, 'black': {}}
                
                if color not in piece_mobility[ply]:
                    piece_mobility[ply][color] = {}
                    
                piece_mobility[ply][color][piece_type] = mobility
            else:
                # 舊格式，黑白共用
                if ply not in piece_mobility:
                    piece_mobility[ply] = {'white': {}, 'black': {}}
                
                piece_mobility[ply]['white'][piece_type] = mobility
                piece_mobility[ply]['black'][piece_type] = mobility
        
        if has_color_column:
            print(f"成功從 CSV 文件載入 {len(piece_mobility)} 個 ply 的分色棋子機動性數據")
        else:
            print(f"成功從 CSV 文件載入 {len(piece_mobility)} 個 ply 的棋子機動性數據（黑白共用）")
        return piece_mobility
        
    except FileNotFoundError:
        print(f"警告: 找不到 CSV 文件 {csv_file_path}，使用預設機動性")
        return {6: {'white': {'p': 2.0, 'n': 4.5, 'b': 7.0, 'r': 8.5, 'q': 15.0},
                   'black': {'p': 2.0, 'n': 4.5, 'b': 7.0, 'r': 8.5, 'q': 15.0}}}
    except Exception as e:
        print(f"載入 CSV 文件時發生錯誤: {e}，使用預設機動性")
        return {6: {'white': {'p': 2.0, 'n': 4.5, 'b': 7.0, 'r': 8.5, 'q': 15.0},
                   'black': {'p': 2.0, 'n': 4.5, 'b': 7.0, 'r': 8.5, 'q': 15.0}}}

# 載入棋子價值和機動性數據
PIECE_THEORETICAL_VALUES = load_piece_values_from_csv()
PIECE_THEORETICAL_MOBILITY = load_piece_mobility_from_csv()

# 為 SEE 邏輯建立一個簡單的子力價值對應表（使用 ply 6 白方作為基準）
default_values = get_piece_values_for_ply(6, PIECE_THEORETICAL_VALUES, 'white')
PIECE_VALUES_FOR_SEE = {
    chess.PAWN: default_values.get('p', 100),
    chess.KNIGHT: default_values.get('n', 320),
    chess.BISHOP: default_values.get('b', 330),
    chess.ROOK: default_values.get('r', 500),
    chess.QUEEN: default_values.get('q', 900),
    chess.KING: 10000
}

# ==============================================================================
# 修正後的靜態交換評估 (SEE) 核心邏輯
# ==============================================================================

def calculate_see(board: chess.Board, move: chess.Move) -> int:
    """
    計算靜態交換評估 (Static Exchange Evaluation)
    完整推演交換序列直到沒有有利的交換為止
    返回交換後的分數差異 (正數表示有利，負數表示不利)
    """
    # 創建棋盤副本用於推演
    board_copy = board.copy()
    target_square = move.to_square
    
    # 記錄交換序列的收益
    gains = []
    
    # 初始移動
    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square) if board.is_capture(move) else None
    
    # 初始收益：被吃棋子的價值
    initial_gain = PIECE_VALUES_FOR_SEE[captured_piece.piece_type] if captured_piece else 0
    gains.append(initial_gain)
    
    # 執行初始移動
    board_copy.push(move)
    
    # 現在目標格子上是剛移動的棋子
    current_piece_on_square = moving_piece
    current_side = board_copy.turn  # 現在輪到對方
    
    # 持續推演交換序列
    while True:
        # 找到當前方所有攻擊目標格子的棋子
        attackers = get_attackers_by_value(board_copy, current_side, target_square)
        
        if not attackers:
            # 沒有攻擊者，交換結束
            break
        
        # 使用最便宜的攻擊者
        cheapest_attacker_square, cheapest_attacker_piece = attackers[0]
        
        # 計算這次交換的收益
        gain = PIECE_VALUES_FOR_SEE[current_piece_on_square.piece_type]
        gains.append(gain)
        
        # 執行交換移動
        exchange_move = chess.Move(cheapest_attacker_square, target_square)
        
        # 檢查移動是否合法（可能被牽制）
        if exchange_move not in board_copy.legal_moves:
            # 這個攻擊者被牽制，嘗試下一個
            attackers.pop(0)
            if not attackers:
                break
            cheapest_attacker_square, cheapest_attacker_piece = attackers[0]
            exchange_move = chess.Move(cheapest_attacker_square, target_square)
            if exchange_move not in board_copy.legal_moves:
                break
        
        board_copy.push(exchange_move)
        
        # 更新狀態
        current_piece_on_square = cheapest_attacker_piece
        current_side = board_copy.turn
    
    # 使用極大極小算法計算最終分數
    return calculate_exchange_value(gains)


def get_attackers_by_value(board: chess.Board, color: chess.Color, square: chess.Square) -> list:
    """
    獲取指定顏色攻擊指定格子的所有棋子，按價值從小到大排序
    返回 [(square, piece), ...] 的列表
    """
    attackers = []
    attacker_squares = board.attackers(color, square)
    
    for attacker_square in attacker_squares:
        piece = board.piece_at(attacker_square)
        if piece and piece.color == color:
            attackers.append((attacker_square, piece))
    
    # 按棋子價值排序（價值小的優先）
    attackers.sort(key=lambda x: PIECE_VALUES_FOR_SEE[x[1].piece_type])
    
    return attackers


def calculate_exchange_value(gains: list) -> int:
    """
    修正的極大極小算法計算交換序列的最終價值
    """
    if not gains:
        return 0
    
    if len(gains) == 1:
        return gains[0]
    
    # 從最後開始向前計算
    # 每一方都會選擇對自己最有利的選項：繼續交換或停止
    value = gains[-1]
    
    for i in range(len(gains) - 2, -1, -1):
        # 當前方選擇：max(停止交換=0, 繼續交換=gains[i]-value)
        value = max(0, gains[i] - value)
    
    return value


def is_piece_pinned(board: chess.Board, square: chess.Square) -> bool:
    """
    檢查指定格子上的棋子是否被牽制
    修正版本：正確處理非走棋方的棋子
    """
    piece = board.piece_at(square)
    if not piece:
        return False
    
    # 創建測試棋盤，移除該棋子
    test_board = board.copy()
    test_board.remove_piece_at(square)
    
    # 找到該顏色的國王位置
    king_square = test_board.king(piece.color)
    if king_square is None:
        return False
    
    # 檢查移除該棋子後，國王是否會被攻擊
    return test_board.is_attacked_by(not piece.color, king_square)


def is_safe_move_for_any_side(board: chess.Board, move: chess.Move, piece_color: chess.Color) -> bool:
    """
    判斷任何一方的棋子移動是否安全
    這是關鍵函數：處理非走棋方棋子的安全性判斷
    """
    # 首先檢查棋子是否被牽制
    if is_piece_pinned(board, move.from_square):
        # 被牽制的棋子只能沿著牽制線移動
        piece = board.piece_at(move.from_square)
        king_square = board.king(piece.color)
        
        # 檢查移動是否沿著國王-棋子-攻擊者的直線
        if not is_move_along_pin_line(board, move, king_square):
            return False
    
    # 創建棋盤副本，設定為該棋子的顏色為走棋方
    temp_board = board.copy()
    temp_board.turn = piece_color
    
    # 檢查移動是否在這個設定下合法
    if move not in temp_board.legal_moves:
        return False
    
    # 將軍移動通常認為是安全的
    temp_board_after_move = temp_board.copy()
    temp_board_after_move.push(move)
    if temp_board_after_move.is_check():
        return True
    
    # 城堡移動認為是安全的
    if temp_board.is_castling(move):
        return True
    
    # 兵升變移動需要特殊處理
    if move.promotion:
        see_score = calculate_see_for_hypothetical_move(board, move, piece_color)
        return see_score >= 0
    
    # 使用SEE評估，但需要特殊處理非走棋方的情況
    see_score = calculate_see_for_hypothetical_move(board, move, piece_color)
    return see_score >= 0


def is_move_along_pin_line(board: chess.Board, move: chess.Move, king_square: chess.Square) -> bool:
    """
    檢查移動是否沿著牽制線進行
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # 計算三點是否共線
    def are_squares_aligned(sq1: chess.Square, sq2: chess.Square, sq3: chess.Square) -> bool:
        """檢查三個格子是否在同一條直線（橫線、縱線或對角線）上"""
        file1, rank1 = chess.square_file(sq1), chess.square_rank(sq1)
        file2, rank2 = chess.square_file(sq2), chess.square_rank(sq2)
        file3, rank3 = chess.square_file(sq3), chess.square_rank(sq3)
        
        # 檢查是否在同一橫線
        if rank1 == rank2 == rank3:
            return True
        
        # 檢查是否在同一縱線
        if file1 == file2 == file3:
            return True
        
        # 檢查是否在同一對角線
        if abs(file1 - file2) == abs(rank1 - rank2) and abs(file2 - file3) == abs(rank2 - rank3):
            # 進一步檢查斜率是否相同
            if (file1 - file2) * (rank2 - rank3) == (file2 - file3) * (rank1 - rank2):
                return True
        
        return False
    
    return are_squares_aligned(king_square, from_square, to_square)


def calculate_see_for_hypothetical_move(board: chess.Board, move: chess.Move, moving_piece_color: chess.Color) -> int:
    """
    為假設性移動計算SEE（特別用於非走棋方棋子）
    """
    # 創建棋盤副本
    temp_board = board.copy()
    
    # 如果移動的棋子不是當前走棋方，需要特殊處理
    if moving_piece_color != board.turn:
        # 暫時改變走棋方來執行移動
        temp_board.turn = moving_piece_color
        
        # 檢查移動是否合法
        if move not in temp_board.legal_moves:
            return -10000  # 不合法的移動認為是不安全的
        
        # 執行移動
        temp_board.push(move)
        
        # 現在輪到對方，檢查剛移動的棋子是否安全
        target_square = move.to_square
        moved_piece = temp_board.piece_at(target_square)
        
        # 如果目標位置沒有被對方攻擊，則安全
        if not temp_board.is_attacked_by(temp_board.turn, target_square):
            return 0  # 安全
        
        # 如果被攻擊，計算交換價值
        # 創建一個虛擬的對方吃子移動來計算SEE
        attackers = list(temp_board.attackers(temp_board.turn, target_square))
        if not attackers:
            return 0
        
        # 找最便宜的攻擊者
        cheapest_attacker = None
        cheapest_value = float('inf')
        
        for attacker_square in attackers:
            attacker_piece = temp_board.piece_at(attacker_square)
            attacker_value = PIECE_VALUES_FOR_SEE[attacker_piece.piece_type]
            
            # 檢查這個攻擊者是否能合法地吃掉目標
            virtual_capture = chess.Move(attacker_square, target_square)
            if virtual_capture in temp_board.legal_moves and attacker_value < cheapest_value:
                cheapest_attacker = virtual_capture
                cheapest_value = attacker_value
        
        if cheapest_attacker:
            # 計算如果被最便宜的攻擊者吃掉後的交換序列
            return -calculate_see(temp_board, cheapest_attacker)
        else:
            return 0
    
    else:
        # 正常的走棋方移動，直接使用標準SEE
        return calculate_see(temp_board, move)


def get_safe_moves_for_piece_improved(board: chess.Board, from_square: chess.Square) -> list:
    """
    改進版：獲取指定格子上棋子的所有安全移動
    正確處理非走棋方棋子和牽制情況
    """
    piece = board.piece_at(from_square)
    if not piece:
        return []
    
    # 檢查棋子是否被牽制
    if is_piece_pinned(board, from_square):
        # 被牽制的棋子，只能考慮沿牽制線的移動
        king_square = board.king(piece.color)
        
        # 創建暫時棋盤，設定為該棋子的顏色回合
        temp_board = board.copy()
        temp_board.turn = piece.color
        
        # 獲取該棋子的所有理論移動
        piece_legal_moves = [
            move for move in temp_board.legal_moves
            if move.from_square == from_square
        ]
        
        # 過濾出沿牽制線的移動
        valid_moves = []
        for move in piece_legal_moves:
            if is_move_along_pin_line(board, move, king_square):
                # 進一步檢查移動後國王是否安全
                test_board = temp_board.copy()
                test_board.push(move)
                if not test_board.is_check():
                    valid_moves.append(move.uci())
        
        return valid_moves
    
    else:
        # 非牽制棋子，使用標準安全性檢查
        # 創建暫時棋盤，設定為該棋子的顏色回合
        temp_board = board.copy()
        temp_board.turn = piece.color
        
        # 獲取該棋子在假設為其回合時的所有合法移動
        piece_legal_moves = [
            move for move in temp_board.legal_moves
            if move.from_square == from_square
        ]
        
        # 過濾出安全的移動
        safe_moves = []
        for move in piece_legal_moves:
            if is_safe_move_for_any_side(board, move, piece.color):
                safe_moves.append(move.uci())
        
        return safe_moves


def analyze_piece_mobility_and_safety(board: chess.Board, square: chess.Square) -> dict:
    """
    分析指定格子上棋子的機動性和安全性
    返回詳細分析結果
    """
    piece = board.piece_at(square)
    if not piece:
        return {"error": "No piece on this square"}
    
    # 檢查是否被牽制
    is_pinned = is_piece_pinned(board, square)
    
    # 創建假設該棋子可以移動的棋盤
    temp_board = board.copy()
    temp_board.turn = piece.color
    
    # 獲取所有理論上的合法移動
    all_theoretical_moves = [
        move for move in temp_board.legal_moves
        if move.from_square == square
    ]
    
    # 分析每個移動的安全性
    move_analysis = []
    safe_moves = []
    unsafe_moves = []
    
    for move in all_theoretical_moves:
        is_safe = is_safe_move_for_any_side(board, move, piece.color)
        see_score = calculate_see_for_hypothetical_move(board, move, piece.color)
        
        move_info = {
            "move": move.uci(),
            "to_square": chess.square_name(move.to_square),
            "is_capture": temp_board.is_capture(move),
            "is_safe": is_safe,
            "see_score": see_score,
            "gives_check": False
        }
        
        # 檢查是否將軍
        test_board = temp_board.copy()
        test_board.push(move)
        move_info["gives_check"] = test_board.is_check()
        
        move_analysis.append(move_info)
        
        if is_safe:
            safe_moves.append(move.uci())
        else:
            unsafe_moves.append(move.uci())
    
    return {
        "piece": piece.symbol(),
        "square": chess.square_name(square),
        "is_current_player": piece.color == board.turn,
        "is_pinned": is_pinned,
        "total_legal_moves": len(all_theoretical_moves),
        "safe_moves_count": len(safe_moves),
        "unsafe_moves_count": len(unsafe_moves),
        "safe_moves": safe_moves,
        "unsafe_moves": unsafe_moves,
        "detailed_analysis": move_analysis,
        "mobility_score": len(safe_moves) / max(1, len(all_theoretical_moves))  # 0-1之間的機動性分數
    }


def is_safe_move(board: chess.Board, move: chess.Move) -> bool:
    """
    原有函數的兼容性包裝
    """
    moving_piece = board.piece_at(move.from_square)
    if not moving_piece:
        return False
    
    return is_safe_move_for_any_side(board, move, moving_piece.color)


def get_safe_moves_for_piece(board: chess.Board, from_square: chess.Square) -> list:
    """
    原有函數的兼容性包裝，使用改進的邏輯
    """
    return get_safe_moves_for_piece_improved(board, from_square)

# === Pawn feature helpers =====================================================

def _has_opp_pawn_in_front_on_file(board: chess.Board, file_idx: int, start_rank: int, direction: int, opp_color: chess.Color) -> bool:
    r = start_rank + direction
    while 0 <= r <= 7:
        sq = chess.square(file_idx, r)
        p = board.piece_at(sq)
        if p and p.color == opp_color and p.piece_type == chess.PAWN:
            return True
        r += direction
    return False

def is_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    file_idx = chess.square_file(square)
    rank = chess.square_rank(square)
    direction = 1 if color == chess.WHITE else -1
    opp = not color
    for df in (-1, 0, 1):
        f = file_idx + df
        if 0 <= f <= 7:
            if _has_opp_pawn_in_front_on_file(board, f, rank, direction, opp):
                return False
    return True

def is_candidate_passed(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    # 簡化版候選通路兵：同檔前方沒有對方兵，且左右檔前方最多只被一側的對方兵阻擋
    if is_passed_pawn(board, square, color):
        return False
    file_idx = chess.square_file(square)
    rank = chess.square_rank(square)
    direction = 1 if color == chess.WHITE else -1
    opp = not color

    if _has_opp_pawn_in_front_on_file(board, file_idx, rank, direction, opp):
        return False

    blockers = 0
    for df in (-1, 1):
        f = file_idx + df
        if 0 <= f <= 7 and _has_opp_pawn_in_front_on_file(board, f, rank, direction, opp):
            blockers += 1
    return blockers == 1

def distance_to_promotion(square: chess.Square, color: chess.Color) -> int:
    rank = chess.square_rank(square)
    return (7 - rank) if color == chess.WHITE else rank

def is_shielding_high_value_piece(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    """
    判斷該兵是否作為「盾牌」保護己方重要子力（王/后/車）。
    做法：移除兵後，是否使任一己方王/后/車從「未被攻擊」→「被攻擊」。
    """
    piece = board.piece_at(square)
    if not piece or piece.piece_type != chess.PAWN:
        return False

    opp = not color
    # 先記錄原來是否被攻擊
    critical_squares = [sq for sq, pc in board.piece_map().items()
                        if pc.color == color and pc.piece_type in (chess.KING, chess.QUEEN, chess.ROOK)]
    before = {sq: board.is_attacked_by(opp, sq) for sq in critical_squares}

    temp = board.copy()
    temp.remove_piece_at(square)
    after = {sq: temp.is_attacked_by(opp, sq) for sq in critical_squares}

    # 只要有任一要角從未被攻擊 → 被攻擊，即視為盾牌
    return any((not before[sq]) and after[sq] for sq in critical_squares)

def count_attackers_defenders(board: chess.Board, square: chess.Square, color: chess.Color) -> tuple[int, int]:
    opp = not color
    attackers = len(board.attackers(opp, square))
    defenders = len(board.attackers(color, square))
    return attackers, defenders

def get_pawn_safe_pushes(board: chess.Board, square: chess.Square, safe_moves_uci: list[str], color: chess.Color) -> int:
    """
    回傳該兵「安全」直進步數（不含吃子），0/1/2。
    以 safe_moves_uci（SEE 過濾後）為準。
    """
    pushes = 0
    one_rank = chess.square_rank(square) + (1 if color == chess.WHITE else -1)
    if 0 <= one_rank <= 7:
        sq = chess.square(chess.square_file(square), one_rank)
        m1 = chess.Move(square, sq).uci()
        if m1 in safe_moves_uci:
            pushes += 1
            # 兩步起步
            start_rank = 1 if color == chess.WHITE else 6
            if chess.square_rank(square) == start_rank:
                two_rank = chess.square_rank(square) + (2 if color == chess.WHITE else -2)
                two_sq = chess.square(chess.square_file(square), two_rank)
                m2 = chess.Move(square, two_sq).uci()
                if m2 in safe_moves_uci:
                    pushes += 1
    return pushes

# ==============================================================================
# 4. 核心功能函式 (部分函式已更新)
# ==============================================================================
def analyze_position_deeply(fen: str):
    if not os.path.exists(STOCKFISH_PATH): raise FileNotFoundError(f"Stockfish not found at {STOCKFISH_PATH}")
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        board = chess.Board(fen)

        # KMAPS Evaluation
        kmap_eval = KMAPS(fen)
        k_eval = kmap_eval.evaluate_king_safety() * 100
        m_eval = kmap_eval.evaluate_material() * 100
        a_eval = kmap_eval.evaluate_piece_activity() * 100
        p_eval = kmap_eval.evaluate_pawn_structure() * 100
        s_eval = kmap_eval.evaluate_space() * 100
        kmap_scores = {"K": k_eval, "M": m_eval, "A": a_eval, "P": p_eval, "S": s_eval}
        print("--- KMAPS Evaluation ---")
        print(f"  > K: {k_eval:.0f}, M: {m_eval:.0f}, A: {a_eval:.0f}, P: {p_eval:.0f}, S: {s_eval:.0f}")

        print("--- Analyzing initial position ---")
        info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
        initial_eval = get_eval_from_info(info)
        current_ply = get_current_ply(board)
        game_phase = get_game_phase_name(board)
        theoretical_values_white = get_piece_values_for_ply(current_ply, PIECE_THEORETICAL_VALUES, 'white')
        theoretical_values_black = get_piece_values_for_ply(current_ply, PIECE_THEORETICAL_VALUES, 'black')
        theoretical_mobility_white = get_piece_mobility_for_ply(current_ply, PIECE_THEORETICAL_MOBILITY, 'white')
        theoretical_mobility_black = get_piece_mobility_for_ply(current_ply, PIECE_THEORETICAL_MOBILITY, 'black')
        piece_values = []
        print(f"Ply: {current_ply} ({game_phase}), Eval: {initial_eval/100.0:.2f}")
        print("--- Calculating piece values, mobility and safe moves (color-separated) ---")

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece or piece.piece_type == chess.KING: continue
            
            # 修正：根據棋子符號的大小寫判斷顏色（大寫=白方，小寫=黑方）
            is_white_piece = piece.symbol().isupper()
            color_str = 'white' if is_white_piece else 'black'
            theoretical_values = theoretical_values_white if is_white_piece else theoretical_values_black
            theoretical_mobility = theoretical_mobility_white if is_white_piece else theoretical_mobility_black
            
            # 檢查是否被牽制
            if is_piece_pinned(board, square):
                piece_safe_moves = get_safe_moves_for_piece(board, square)
                piece_info = {
                    "piece": piece.symbol(), "square": chess.square_name(square),
                    "actual_value": "pinned", "theoretical_value": theoretical_values.get(piece.symbol().lower(), 0),
                    "actual_mobility": len(piece_safe_moves), "theoretical_mobility": theoretical_mobility.get(piece.symbol().lower(), 0),
                    "safe_moves": piece_safe_moves, "color": color_str
                }
                piece_values.append(piece_info)
                print(f"  > {piece.symbol()}{piece_info['square']} ({color_str}): PINNED, {len(piece_safe_moves)} safe moves (theory: {piece_info['theoretical_mobility']:.1f})")
                continue
            
            piece_safe_moves = get_safe_moves_for_piece(board, square)
            
            # === 兵的結構特徵（只對兵計算） ===
            pawn_features = None
            if piece.piece_type == chess.PAWN:
                passed_ = is_passed_pawn(board, square, piece.color)
                cand_passed = is_candidate_passed(board, square, piece.color)
                attackers, defenders = count_attackers_defenders(board, square, piece.color)
                protected_passed = passed_ and defenders >= attackers
                shielding = is_shielding_high_value_piece(board, square, piece.color)
                dist = distance_to_promotion(square, piece.color)
                safe_pushes = get_pawn_safe_pushes(board, square, piece_safe_moves, piece.color)

                pawn_features = {
                    "is_passed": passed_,
                    "is_candidate_passed": cand_passed,
                    "is_protected_passed": protected_passed,
                    "is_shielding_high_value_piece": shielding,
                    "distance_to_promotion": dist,
                    "safe_pushes": safe_pushes,                       # 0/1/2
                    "defenders_more_than_attackers": defenders > attackers
                }
            
            temp_board = board.copy()
            piece_symbol = piece.symbol()
            temp_board.remove_piece_at(square)
            value = "error"
            if not temp_board.is_check():
                info = engine.analyse(temp_board, chess.engine.Limit(depth=DEPTH-1))
                new_eval = get_eval_from_info(info)
                value = (initial_eval - new_eval) if is_white_piece else (new_eval - initial_eval)

            piece_info = {
                "piece": piece_symbol, "square": chess.square_name(square),
                "actual_value": value, "theoretical_value": theoretical_values.get(piece.symbol().lower(), 0),
                "actual_mobility": len(piece_safe_moves), "theoretical_mobility": theoretical_mobility.get(piece.symbol().lower(), 0),
                "safe_moves": piece_safe_moves, "color": color_str
            }
            if pawn_features is not None:
                piece_info["pawn_features"] = pawn_features
            
            piece_values.append(piece_info)
            mobility_comparison = f"mobility: {len(piece_safe_moves)} vs theory: {piece_info['theoretical_mobility']:.1f}"
            print(f"  > {piece_symbol}{piece_info['square']} ({color_str}): {value} (theory: {piece_info['theoretical_value']}), {mobility_comparison}")

        return {"fen": fen, "initial_eval": initial_eval, "game_phase": game_phase, "current_ply": current_ply, "piece_values": piece_values, "kmap_scores": kmap_scores}
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return None
    finally:
        if engine: engine.quit()


# ==============================================================================
# 5. 主程式執行
# ==============================================================================
if __name__ == "__main__":
    fen_string = "3r2k1/p1q2pp1/1p5p/1Qb5/4PPbP/2R3P1/P2rN1B1/4RK2 b - - 0 29"
    start_time = time.time()
    analysis_result = analyze_position_deeply(fen_string)
    
    if analysis_result:
        structured_output = get_llm_analysis(analysis_result, API_KEY, MODEL)
        if structured_output:
            print("\n\n==================================================")
            print("         西洋棋 AI 局面快評")
            print("==================================================")
            print(f"FEN: {fen_string}\n")

            print(f"[*] 總體形勢: {structured_output.overall_assessment}\n")
            
            print("--- 白方 (White) ---")
            print("  [+] 正面點評:")
            for point in structured_output.white_positive_points: print(f"    - {point}")
            print("  [-] 負面點評:")
            for point in structured_output.white_negative_points: print(f"    - {point}")
            print()

            print("--- 黑方 (Black) ---")
            print("  [+] 正面點評:")
            for point in structured_output.black_positive_points: print(f"    - {point}")
            print("  [-] 負面點評:")
            for point in structured_output.black_negative_points: print(f"    - {point}")
            print("\n--------------------------------------------------")

    end_time = time.time()
    print(f"\n分析完成，總耗時: {end_time - start_time:.2f} 秒")
