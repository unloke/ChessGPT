import re
import math
from typing import Dict, List, Optional, Set, Tuple


class KMAPS:
    """
    Chess position evaluator implementing the K-MAPS system:
    K - King safety
    M - Material
    A - Activity
    P - Pawn structure
    S - Space
    """
    
    def __init__(self, fen: str):
        """Initialize the evaluator with a FEN string."""
        self.board = self.parse_fen(fen)
        self.moves_cache = {}
        self.piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
        self.central_squares = [27, 28, 35, 36]  # d4, d5, e4, e5
    
    def parse_fen(self, fen: str) -> List[Optional[str]]:
        """Parse FEN string to 8x8 board array."""
        board = [None] * 64
        position = fen.split(' ')[0]
        square = 0
        
        for char in position:
            if square >= len(board):
                break
            if char == '/':
                continue
            if char.isdigit():
                square += int(char)
            else:
                board[square] = char
                square += 1
        
        return board
    
    def normalize(self, score: float, min_val: float, max_val: float, area: str) -> float:
        """Normalize a score to [-1, 1]."""
        if score >= max_val:
            return 1.0
        if score <= min_val:
            return -1.0
        normalized = (2 * (score - min_val)) / (max_val - min_val) - 1
        return normalized
    
    def evaluate_king_safety(self) -> float:
        """Evaluate King Safety (K)."""
        white_score = 0.0
        black_score = 0.0
        
        try:
            white_king_idx = self.board.index('K')
        except ValueError:
            white_king_idx = -1
        
        try:
            black_king_idx = self.board.index('k')
        except ValueError:
            black_king_idx = -1
        
        if white_king_idx == -1 or black_king_idx == -1:
            return 0.0
        
        # Pawn shield
        white_pawn_shield = self.get_pawn_shield(white_king_idx, True)
        black_pawn_shield = self.get_pawn_shield(black_king_idx, False)
        white_score += white_pawn_shield * 2
        black_score += black_pawn_shield * 2
        
        # Open files near king
        white_open_files = self.get_open_files_near_king(white_king_idx, True)
        black_open_files = self.get_open_files_near_king(black_king_idx, False)
        white_score -= white_open_files * 1.5
        black_score -= black_open_files * 1.5
        
        # Enemy pieces near king
        white_threats = self.get_enemy_threats(white_king_idx, True)
        black_threats = self.get_enemy_threats(black_king_idx, False)
        white_score -= white_threats * 1
        black_score -= black_threats * 1
        
        # Hole detection
        white_holes = self.get_king_holes(white_king_idx, True)
        black_holes = self.get_king_holes(black_king_idx, False)
        white_score -= white_holes * 0.5
        black_score -= black_holes * 0.5
        
        diff = white_score - black_score
        return self.normalize(diff, -15, 15, 'K')
    
    def get_pawn_shield(self, king_idx: int, is_white: bool) -> float:
        """Calculate pawn shield strength."""
        rank = king_idx // 8
        file = king_idx % 8
        score = 0.0
        offsets = [-9, -8, -7] if is_white else [7, 8, 9]
        
        for offset in offsets:
            idx = king_idx + offset
            if 0 <= idx < 64 and abs((idx % 8) - file) <= 1:
                piece = self.board[idx]
                if (is_white and piece == 'P') or (not is_white and piece == 'p'):
                    score += 1
        
        return score
    
    def get_open_files_near_king(self, king_idx: int, is_white: bool) -> int:
        """Count open files near the king."""
        file = king_idx % 8
        open_files = 0
        
        for f in range(max(0, file - 1), min(8, file + 2)):
            has_pawn = False
            for r in range(8):
                idx = r * 8 + f
                if (is_white and self.board[idx] == 'P') or (not is_white and self.board[idx] == 'p'):
                    has_pawn = True
                    break
            if not has_pawn:
                open_files += 1
        
        return open_files
    
    def get_enemy_threats(self, king_idx: int, is_white: bool) -> float:
        """Calculate enemy piece threats near the king."""
        file = king_idx % 8
        rank = king_idx // 8
        threats = 0.0
        
        for r in range(max(0, rank - 2), min(8, rank + 3)):
            for f in range(max(0, file - 2), min(8, file + 3)):
                idx = r * 8 + f
                piece = self.board[idx]
                if piece and piece not in ['K', 'k']:
                    is_enemy = piece.islower() if is_white else piece.isupper()
                    if is_enemy:
                        threats += self.piece_values.get(piece.lower(), 0)
        
        return threats / 10
    
    def get_king_holes(self, king_idx: int, is_white: bool) -> int:
        """Count holes around the king that can never be attacked by friendly pawns."""
        file = king_idx % 8
        rank = king_idx // 8
        holes = 0
        
        for r in range(max(0, rank - 1), min(8, rank + 2)):
            for f in range(max(0, file - 1), min(8, file + 2)):
                if r == rank and f == file:
                    continue
                if is_white and r <= 1:
                    continue
                if not is_white and r >= 6:
                    continue
                idx = r * 8 + f
                if not self.can_ever_be_attacked_by_pawn(idx, is_white):
                    holes += 1
        
        return holes
    
    def can_ever_be_attacked_by_pawn(self, idx: int, is_white: bool) -> bool:
        """Check if a square can ever be attacked by a friendly pawn."""
        file = idx % 8
        rank = idx // 8
        pawn_rank = rank - 1 if is_white else rank + 1
        
        if pawn_rank < 0 or pawn_rank > 7:
            return False
        
        friendly_pawn = 'P' if is_white else 'p'
        for f in range(max(0, file - 1), min(8, file + 2)):
            if f == file:
                continue
            pawn_idx = pawn_rank * 8 + f
            if self.board[pawn_idx] == friendly_pawn:
                return True
        
        return False
    
    def evaluate_material(self) -> float:
        """Evaluate Material (M)."""
        white_material = 0.0
        black_material = 0.0
        white_bishops = 0
        black_bishops = 0
        white_knights = 0
        black_knights = 0
        white_rooks = 0
        black_rooks = 0
        
        for piece in self.board:
            if not piece:
                continue
            
            # Count specific pieces
            if piece == 'B':
                white_bishops += 1
            elif piece == 'b':
                black_bishops += 1
            elif piece == 'N':
                white_knights += 1
            elif piece == 'n':
                black_knights += 1
            elif piece == 'R':
                white_rooks += 1
            elif piece == 'r':
                black_rooks += 1
            
            value = self.piece_values.get(piece.lower(), 0)
            if piece.isupper():
                white_material += value
            else:
                black_material += value
        
        # Bishop pair bonus
        if white_bishops == 2:
            white_material += 0.3
        if black_bishops == 2:
            black_material += 0.3
        
        # Minor piece vs. rook imbalance penalty
        if white_rooks == 1 and white_bishops + white_knights >= 2 and black_rooks == 0:
            white_material -= 0.2
        if black_rooks == 1 and black_bishops + black_knights >= 2 and white_rooks == 0:
            black_material -= 0.2
        
        diff = white_material - black_material
        return self.normalize(diff, -10, 10, 'M')
    
    def evaluate_piece_activity(self) -> float:
        """Evaluate Piece Activity (A)."""
        white_score = 0.0
        black_score = 0.0
        
        # Mobility
        for i in range(64):
            piece = self.board[i]
            if not piece:
                continue
            is_white = piece.isupper()
            moves = self.get_pseudo_legal_moves(i, piece)
            if is_white:
                white_score += len(moves) * 0.1
            else:
                black_score += len(moves) * 0.1
        
        # Bonus for rooks and queens on open files
        for i in range(64):
            piece = self.board[i]
            if not piece:
                continue
            is_white = piece.isupper()
            piece_type = piece.lower()
            if piece_type in ['r', 'q']:
                file = i % 8
                if self.is_open_file(file):
                    if is_white:
                        white_score += 0.3
                    else:
                        black_score += 0.3
        
        # Knight outpost bonus
        outpost_squares = [19, 20, 27, 28, 35, 36, 43, 44]
        for i in range(64):
            piece = self.board[i]
            if not piece or piece.lower() != 'n':
                continue
            is_white = piece.isupper()
            rank = i // 8
            file = i % 8
            if i in outpost_squares:
                if is_white:
                    pawn_support = ((file > 0 and self.board[(rank + 1) * 8 + file - 1] == 'P') or
                                  (file < 7 and self.board[(rank + 1) * 8 + file + 1] == 'P'))
                else:
                    pawn_support = ((file > 0 and self.board[(rank - 1) * 8 + file - 1] == 'p') or
                                  (file < 7 and self.board[(rank - 1) * 8 + file + 1] == 'p'))
                
                enemy_pawn_attack = self.can_ever_be_attacked_by_pawn(i, not is_white)
                if not enemy_pawn_attack:
                    bonus = 0.3 if pawn_support else 0.15
                    if is_white:
                        white_score += bonus
                    else:
                        black_score += bonus
        
        # Central control
        for center_idx in self.central_squares:
            white_attacks = 0
            black_attacks = 0
            for i in range(64):
                piece = self.board[i]
                if not piece:
                    continue
                is_white_piece = piece.isupper()
                if piece.lower() == 'p':
                    attacks = self.get_pawn_attacks(i, is_white_piece)
                    if center_idx in attacks:
                        if is_white_piece:
                            white_attacks += 1
                        else:
                            black_attacks += 1
                else:
                    moves = self.get_pseudo_legal_moves(i, piece)
                    if center_idx in moves:
                        if is_white_piece:
                            white_attacks += 1
                        else:
                            black_attacks += 1
            
            if white_attacks > black_attacks:
                white_score += 0.5
            elif black_attacks > white_attacks:
                black_score += 0.5
        
        # Diagonal control
        long_diagonals = [
            [0, 9, 18, 27, 36, 45, 54, 63],  # a1-h8
            [7, 14, 21, 28, 35, 42, 49, 56]  # a8-h1
        ]
        
        for diagonal in long_diagonals:
            white_attacks = 0
            black_attacks = 0
            white_piece_on_diagonal = False
            black_piece_on_diagonal = False
            
            # Check for pieces directly on the diagonal
            for idx in diagonal:
                piece = self.board[idx]
                if not piece:
                    continue
                is_white = piece.isupper()
                piece_type = piece.lower()
                if piece_type in ['b', 'q']:
                    if is_white:
                        white_piece_on_diagonal = True
                    else:
                        black_piece_on_diagonal = True
            
            # Check for pieces controlling the diagonal
            for i in range(64):
                piece = self.board[i]
                if not piece or piece.lower() not in ['b', 'q']:
                    continue
                is_white = piece.isupper()
                moves = self.get_pseudo_legal_moves(i, piece, control=True)
                for idx in diagonal:
                    if idx in moves:
                        if is_white:
                            white_attacks += 1
                        else:
                            black_attacks += 1
            
            # Scoring
            if white_piece_on_diagonal:
                white_score += 0.3
            if black_piece_on_diagonal:
                black_score += 0.3
            white_score += white_attacks * 0.1
            black_score += black_attacks * 0.1
        
        # Threats
        for i in range(64):
            piece = self.board[i]
            if not piece:
                continue
            is_white = piece.isupper()
            if piece.lower() == 'p':
                moves = self.get_pawn_attacks(i, is_white) + self.get_pseudo_legal_moves(i, piece)
            else:
                moves = self.get_pseudo_legal_moves(i, piece)
            
            opp_king = 'k' if is_white else 'K'
            try:
                opp_king_idx = self.board.index(opp_king)
            except ValueError:
                opp_king_idx = -1
            
            for move in moves:
                target_piece = self.board[move] if 0 <= move < 64 else None
                
                # Check for checks
                if move == opp_king_idx:
                    if is_white:
                        white_score += 0.3
                    else:
                        black_score += 0.3
                
                # Check for captures
                if target_piece:
                    is_enemy = target_piece.islower() if is_white else target_piece.isupper()
                    if is_enemy:
                        value = self.piece_values.get(target_piece.lower(), 0)
                        if is_white:
                            white_score += value * 0.1
                        else:
                            black_score += value * 0.1
                
                # Promotion threats
                if piece.lower() == 'p':
                    rank = i // 8
                    file = i % 8
                    direction = -1 if is_white else 1
                    next_idx = (rank + direction) * 8 + file
                    if ((is_white and rank == 1) or (not is_white and rank == 6)):
                        if (not self.board[next_idx] or 
                            (self.board[next_idx] and 
                             ((is_white and self.board[next_idx].islower()) or
                              (not is_white and self.board[next_idx].isupper())))):
                            if is_white:
                                white_score += 0.4
                            else:
                                black_score += 0.4
        
        diff = white_score - black_score
        return self.normalize(diff, -15, 15, 'A')
    
    def get_pawn_attacks(self, idx: int, is_white: bool) -> List[int]:
        """Get pawn attack squares."""
        file = idx % 8
        rank = idx // 8
        attacks = []
        direction = -1 if is_white else 1
        
        for f_offset in [-1, 1]:
            new_file = file + f_offset
            new_rank = rank + direction
            if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                attack_idx = new_rank * 8 + new_file
                attacks.append(attack_idx)
        
        return attacks
    
    def get_pseudo_legal_moves(self, idx: int, piece: str, control: bool = False) -> List[int]:
        """Get pseudo-legal moves for a piece."""
        cache_key = f"{piece}{idx}"
        if cache_key in self.moves_cache:
            return self.moves_cache[cache_key]
        
        moves = []
        file = idx % 8
        rank = idx // 8
        is_white = piece.isupper()
        piece_type = piece.lower()
        
        directions = {
            'r': [[0, 1], [0, -1], [1, 0], [-1, 0]],
            'b': [[1, 1], [1, -1], [-1, 1], [-1, -1]],
            'q': [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]],
            'n': [[2, 1], [2, -1], [-2, 1], [-2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2]],
            'k': [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]],
        }
        
        if piece_type == 'p':
            if control:
                raise ValueError("Should not use control=True for pawns")
            direction = -1 if is_white else 1
            start_rank = 6 if is_white else 1
            
            # Forward move
            new_idx = idx + direction * 8
            if 0 <= new_idx < 64 and not self.board[new_idx]:
                moves.append(new_idx)
                # Double move from starting rank
                if rank == start_rank:
                    new_idx += direction * 8
                    if not self.board[new_idx]:
                        moves.append(new_idx)
            
            # Captures
            for f in [-1, 1]:
                new_idx = idx + direction * 8 + f
                if (0 <= new_idx < 64 and abs((new_idx % 8) - file) == 1):
                    target = self.board[new_idx]
                    if target:
                        is_enemy = target.islower() if is_white else target.isupper()
                        if is_enemy:
                            moves.append(new_idx)
        
        elif piece_type in directions:
            for dr, df in directions[piece_type]:
                r = rank + dr
                f = file + df
                steps = 1 if piece_type in ['n', 'k'] else 7
                
                while steps > 0 and 0 <= r < 8 and 0 <= f < 8:
                    new_idx = r * 8 + f
                    target = self.board[new_idx]
                    if target:
                        target_white = target.isupper()
                        if target_white == is_white and not control:
                            break
                        moves.append(new_idx)
                        break
                    moves.append(new_idx)
                    r += dr
                    f += df
                    steps -= 1
        
        result = [i for i in moves if 0 <= i < 64]
        self.moves_cache[cache_key] = result
        return result
    
    def is_open_file(self, file: int) -> bool:
        """Check if a file is open (no pawns)."""
        for r in range(8):
            idx = r * 8 + file
            if self.board[idx] in ['P', 'p']:
                return False
        return True
    
    def evaluate_pawn_structure(self) -> float:
        """Evaluate Pawn Structure (P)."""
        white_score = 0.0
        black_score = 0.0
        
        # Count pawns per file
        white_pawns = [0] * 8
        black_pawns = [0] * 8
        for i in range(64):
            if self.board[i] == 'P':
                white_pawns[i % 8] += 1
            if self.board[i] == 'p':
                black_pawns[i % 8] += 1
        
        # Doubled pawns
        for f in range(8):
            if white_pawns[f] > 1:
                white_score -= (white_pawns[f] - 1) * 0.5
            if black_pawns[f] > 1:
                black_score -= (black_pawns[f] - 1) * 0.5
        
        # Isolated pawns
        for f in range(8):
            if white_pawns[f]:
                left_support = white_pawns[max(0, f - 1)] if f > 0 else 0
                right_support = white_pawns[min(7, f + 1)] if f < 7 else 0
                if not left_support and not right_support:
                    white_score -= 0.5
            
            if black_pawns[f]:
                left_support = black_pawns[max(0, f - 1)] if f > 0 else 0
                right_support = black_pawns[min(7, f + 1)] if f < 7 else 0
                if not left_support and not right_support:
                    black_score -= 0.5
        
        # Passed pawns
        for i in range(64):
            piece = self.board[i]
            if piece == 'P':
                if self.is_passed_pawn(i, True):
                    r = i // 8
                    advanced_rows = 6 - r
                    white_score += 0.2 * advanced_rows
            elif piece == 'p':
                if self.is_passed_pawn(i, False):
                    r = i // 8
                    advanced_rows = r - 1
                    black_score += 0.2 * advanced_rows
        
        # Pawn islands
        white_islands = self.count_pawn_islands(white_pawns)
        black_islands = self.count_pawn_islands(black_pawns)
        white_score -= (white_islands - 1) * 0.5
        black_score -= (black_islands - 1) * 0.5
        
        # Backward pawns
        for i in range(64):
            piece = self.board[i]
            if piece == 'P' and self.is_backward_pawn(i, True):
                penalty = 0.5 if self.is_open_file(i % 8) else 0.3
                white_score -= penalty
            if piece == 'p' and self.is_backward_pawn(i, False):
                penalty = 0.5 if self.is_open_file(i % 8) else 0.3
                black_score -= penalty
        
        # Pawn chain bonus
        for i in range(64):
            piece = self.board[i]
            if not piece or piece.lower() != 'p':
                continue
            is_white = piece == 'P'
            file = i % 8
            rank = i // 8
            support_offsets = [9, 7] if is_white else [-9, -7]
            
            for offset in support_offsets:
                support_idx = i + offset
                if (0 <= support_idx < 64 and 
                    abs((support_idx % 8) - file) == 1 and
                    self.board[support_idx] == piece):
                    if is_white:
                        white_score += 0.1
                    else:
                        black_score += 0.1
        
        diff = white_score - black_score
        return self.normalize(diff, -7, 7, 'P')
    
    def is_passed_pawn(self, idx: int, is_white: bool) -> bool:
        """Check if a pawn is passed."""
        file = idx % 8
        rank = idx // 8
        direction = -1 if is_white else 1
        
        r = rank + direction
        while (is_white and r >= 0) or (not is_white and r < 8):
            for f in range(max(0, file - 1), min(8, file + 2)):
                check_idx = r * 8 + f
                enemy_pawn = 'p' if is_white else 'P'
                if self.board[check_idx] == enemy_pawn:
                    return False
            r += direction
        
        return True
    
    def count_pawn_islands(self, pawns: List[int]) -> int:
        """Count the number of pawn islands."""
        islands = 0
        in_island = False
        for f in range(8):
            if pawns[f] > 0:
                if not in_island:
                    islands += 1
                    in_island = True
            else:
                in_island = False
        return islands
    
    def can_be_captured(self, idx: int, is_white: bool) -> bool:
        """Check if a square can be captured by opponent pieces."""
        opp_pattern = re.compile(r'[a-z]' if is_white else r'[A-Z]')
        for i in range(64):
            piece = self.board[i]
            if not piece or not opp_pattern.match(piece):
                continue
            is_pawn = piece.lower() == 'p'
            moves = (self.get_pawn_attacks(i, not is_white) if is_pawn 
                    else self.get_pseudo_legal_moves(i, piece))
            if idx in moves:
                return True
        return False
    
    def is_backward_pawn(self, idx: int, is_white: bool) -> bool:
        """Check if a pawn is backward."""
        file = idx % 8
        rank = idx // 8
        direction = -1 if is_white else 1
        next_rank = rank + direction
        
        if next_rank < 0 or next_rank > 7:
            return False
        
        forward_idx = next_rank * 8 + file
        if self.board[forward_idx]:  # Blocked
            return False
        
        if self.can_be_captured(forward_idx, is_white):
            friendly_pawn = 'P' if is_white else 'p'
            has_support = ((file > 0 and self.board[rank * 8 + file - 1] == friendly_pawn) or
                          (file < 7 and self.board[rank * 8 + file + 1] == friendly_pawn))
            return not has_support
        
        return False
    
    def evaluate_space(self) -> float:
        """Evaluate Space Advantage (S)."""
        white_score = 0.0
        black_score = 0.0
        
        # Count controlled squares in opponent's half
        white_controlled = set()
        black_controlled = set()
        
        for i in range(64):
            piece = self.board[i]
            if not piece:
                continue
            is_pawn = piece.lower() == 'p'
            is_white = piece.isupper()
            moves = (self.get_pawn_attacks(i, is_white) if is_pawn 
                    else self.get_pseudo_legal_moves(i, piece, control=True))
            
            for move in moves:
                rank = move // 8
                if is_white and rank <= 3:
                    white_controlled.add(move)
                if not is_white and rank >= 4:
                    black_controlled.add(move)
        
        white_score += len(white_controlled) * 0.1
        black_score += len(black_controlled) * 0.1
        
        # Bonus for central pawns
        for idx in self.central_squares:
            if self.board[idx] == 'P':
                white_score += 1
            if self.board[idx] == 'p':
                black_score += 1
        
        # Penalty for cramped pieces
        for i in range(64):
            piece = self.board[i]
            if not piece or piece.lower() == 'k':
                continue
            is_white = piece.isupper()
            moves = (self.get_pseudo_legal_moves(i, piece) + self.get_pawn_attacks(i, is_white) 
                    if piece.lower() == 'p' 
                    else self.get_pseudo_legal_moves(i, piece))
            
            if len(moves) == 0:
                penalty = self.piece_values.get(piece.lower(), 1)
                cramped_penalty = penalty * 0.1 + 0.2
                if is_white:
                    white_score -= cramped_penalty
                else:
                    black_score -= cramped_penalty
        
        diff = white_score - black_score
        return self.normalize(diff, -7, 7, 'S')
    
    def evaluate(self) -> Dict[str, float]:
        """Main evaluation function returning K-MAPS scores."""
        return {
            'K': self.evaluate_king_safety(),
            'M': self.evaluate_material(),
            'A': self.evaluate_piece_activity(),
            'P': self.evaluate_pawn_structure(),
            'S': self.evaluate_space(),
        }


