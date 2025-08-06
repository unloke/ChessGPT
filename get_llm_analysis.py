import json
import google.generativeai as genai
from typing import List, NamedTuple

class ChessAnalysis(NamedTuple):
    """定義 AI 輸出的結構，內容為簡潔的短評"""
    overall_assessment: str
    white_positive_points: List[str]
    white_negative_points: List[str]
    black_positive_points: List[str]
    black_negative_points: List[str]

def analyze_piece_patterns(piece_data: dict) -> dict:
    piece_type = piece_data['piece'].lower()
    actual_value = piece_data['actual_value']
    theoretical_value = piece_data['theoretical_value']
    actual_mobility = piece_data['actual_mobility']
    theoretical_mobility = piece_data['theoretical_mobility']
    color = piece_data['color']
    square = piece_data['square']

    pawn_features = piece_data.get('pawn_features', {})
    patterns = []

    # 特例處理：牽制或錯誤
    if actual_value == "pinned":
        return {"patterns": [{
            "type": "tactical_weakness",
            "description": "被牽制無法移動",
            "chess_meaning": "成為戰術弱點或保護要角"
        }]}
    if actual_value == "error":
        return {"patterns": [{
            "type": "error",
            "description": "價值計算錯誤"
        }]}

    # 計算比率
    value_ratio = actual_value / max(theoretical_value, 1) if isinstance(actual_value, (int, float)) else 0
    mobility_ratio = actual_mobility / max(theoretical_mobility, 1) if theoretical_mobility > 0 else 0

    # === 兵分析 ===
    if piece_type == 'p':
        is_passed = bool(pawn_features.get("is_passed"))
        is_cand = bool(pawn_features.get("is_candidate_passed"))
        is_protected = bool(pawn_features.get("is_protected_passed") or pawn_features.get("defenders_more_than_attackers"))
        is_shield = bool(pawn_features.get("is_shielding_high_value_piece"))
        dist = pawn_features.get("distance_to_promotion")
        safe_pushes = pawn_features.get("safe_pushes", 0)

        # 盾牌兵：避免誤判成推進
        if is_shield:
            patterns.append({
                "type": "pawn_shield",
                "description": "兵作為要角盾牌",
                "chess_meaning": "移動恐暴露王/后/車，須審慎"
            })

        # 通路兵 / 候選通路兵 的推進語義
        if (value_ratio > 1.2 and actual_mobility >= 1) or is_passed or is_cand:
            if is_passed:
                desc = "通路兵具備推進條件" if safe_pushes else "通路兵但暫受限"
                meaning = "可能推進，對方須嚴防"
                if is_protected:
                    meaning = "受保護的通路兵，難以阻擋"
                if isinstance(dist, int) and dist is not None and dist <= 3:
                    meaning = "臨近升變，威脅迫切" if not is_protected else "受保護且臨近升變，極具威脅"
                patterns.append({
                    "type": "pawn_advance",
                    "description": desc,
                    "chess_meaning": meaning
                })
            elif is_cand:
                patterns.append({
                    "type": "pawn_advance",
                    "description": "候選通路兵可推進",
                    "chess_meaning": "有機會轉化為通路兵"
                })
            elif value_ratio > 1.2 :
                patterns.append({
                    "type": "defensive_pawn",
                    "description": "防守用的兵",
                    "chess_meaning": "正在進行防守或支援"
                })

        # 其他兵訊號（沿用原本語義，微調敘述）
        if value_ratio > 1.5 and actual_mobility <= 1:
            patterns.append({
                "type": "pawn_control",
                "description": "兵控制關鍵格",
                "chess_meaning": "壓制路線或拖住對手"
            })
        if isinstance(actual_value, (int, float)) and actual_value < 0:
            patterns.append({
                "type": "self_blocking_pawn",
                "description": "自家兵阻擋",
                "chess_meaning": "阻擋到己方子力出子路線或是攻擊路線"
            })
        elif value_ratio < 0.5 and isinstance(actual_value, (int, float)) and actual_value < 50:
            patterns.append({
                "type": "pawn_weakness",
                "description": "兵成為弱點",
                "chess_meaning": "可能被鎖定攻擊"
            })
        if actual_mobility == 0:
            patterns.append({
                "type": "blocked_pawn",
                "description": "兵被阻擋",
                "chess_meaning": "前進路線暫不可行"
            })
        if value_ratio < 0.8 and mobility_ratio > 1.5:
            patterns.append({
                "type": "overactive_piece",
                "description": "脫離戰場且無關緊要的兵",
                "chess_meaning": "不須特別注重的一隻兵"

            })

        # 過度推進且無保護的兵
        is_advanced = dist is not None and dist <= 4
        if is_advanced and not is_protected and value_ratio < 0.8:
            patterns.append({
                "type": "overextended_pawn",
                "description": "過度前傾但價值不高",
                "chess_meaning": "可能脫離主力、暴露於攻擊"
            })

    # === 輕子力分析 ===
    elif piece_type in ['n', 'b']:
        if value_ratio > 2.0:
            patterns.append({
                "type": "tactical_threat",
                "description": "具有戰術威脅的子力",
                "chess_meaning": "可能準備攻擊要點或實施戰術打擊"
            })
        elif value_ratio > 1.2:
            if mobility_ratio > 1.2:
                patterns.append({
                    "type": "active_piece",
                    "description": "極度活躍的輕子力",
                    "chess_meaning": "能靈活調動並控制多個關鍵區域"
                })
            elif mobility_ratio < 0.8:
                patterns.append({
                    "type": "strategic_piece",
                    "description": "戰略位置的輕子力",
                    "chess_meaning": "儘管受限，但具有防禦或支持功能"
                })

        if mobility_ratio < 0.3 and actual_mobility <= 2:
            patterns.append({
                "type": "trapped_piece",
                "description": "接近被困的子力",
                "chess_meaning": "移動空間受限，可能難以脫出"
            })

    # === 重子力分析 ===
    elif piece_type in ['r', 'q']:
        if value_ratio > 1.8:
            patterns.append({
                "type": "dominant_piece",
                "description": "具有支配地位的重子力",
                "chess_meaning": "可能控制開放線、對王形成壓力"
            })

        if piece_type == 'r':
            if value_ratio > 2.5:
                patterns.append({
                    "type": "rook_battery",
                    "description": "形成攻擊組合的車",
                    "chess_meaning": "可能配合后或另一車準備進攻"
                })
            if mobility_ratio < 0.4:
                patterns.append({
                    "type": "passive_rook",
                    "description": "被動防守的車",
                    "chess_meaning": "可能滯留後排或未參與戰局"
                })

        elif piece_type == 'q':
            if value_ratio > 2.0 and mobility_ratio > 1.0:
                patterns.append({
                    "type": "queen_attack",
                    "description": "準備發動攻擊的后",
                    "chess_meaning": "多重威脅或支援戰術手段"
                })
            if mobility_ratio < 0.5:
                patterns.append({
                    "type": "restricted_queen",
                    "description": "活動受限的后",
                    "chess_meaning": "可能被兵型壓制或暫無進攻路線"
                })

    # === 通用加權 ===
    if value_ratio > 3.0:
        patterns.append({
            "type": "critical_piece",
            "description": "關鍵子力",
            "chess_meaning": "可能做為關鍵攻擊或防守支撐"
        })
    if value_ratio > 1.5 and mobility_ratio < 0.5:
        patterns.append({
            "type": "strategic_anchor",
            "description": "戰略支點",
            "chess_meaning": "不靈活但對局面有重要影響"})


    # === 整合棋盤格位推論 ===
    file_char = square[0]
    rank_num = int(square[1])
    if file_char in ['d', 'e'] and rank_num in [4, 5] and value_ratio > 1.3:
        patterns.append({"type": "center_control","description": "中央控制","chess_meaning": "占據核心地帶"})
    if ((color == 'white' and rank_num in [1, 2]) or (color == 'black' and rank_num in [7, 8])) and file_char in ['f', 'g', 'h'] and value_ratio > 1.5:
        patterns.append({"type": "king_safety","description": "王翼防護","chess_meaning": "王附近有保護"})

    return {"patterns": patterns}

def get_llm_analysis(analysis_data: dict, api_key: str, model_name: str) -> ChessAnalysis | None:
    if not api_key: raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    eval_score = analysis_data['initial_eval']
    if eval_score > 80: eval_term = "白方優勢"
    elif eval_score > 30: eval_term = "白方略優"
    elif eval_score < -80: eval_term = "黑方優勢"
    elif eval_score < -30: eval_term = "黑方略優"
    else: eval_term = "均勢"

    # === 修正的數據預處理和格式化（修正顏色判斷邏輯）===
    def format_piece_analysis_enhanced(pieces: list, target_color: str) -> str:
        analysis_str = ""
        # 修正：根據 color 字段而不是大小寫來過濾
        filtered_pieces = [p for p in pieces if p['color'] == target_color]

        for p in filtered_pieces:
            actual, theor = p['actual_value'], p['theoretical_value']
            actual_mobility = p['actual_mobility']
            theor_mobility = p['theoretical_mobility']
            safe_moves_count = len(p['safe_moves'])

            # 使用增強的模式分析
            pattern_analysis = analyze_piece_patterns(p)
            patterns = pattern_analysis.get('patterns', [])

            # 1. 基本價值表現評估
            performance_info = ""
            if isinstance(actual, int):
                if actual > theor * 1.2:
                    performance_info = f"表現出色 (價值: {actual} vs 理論: {theor})"
                elif actual < theor * 0.8:
                    performance_info = f"表現不佳 (價值: {actual} vs 理論: {theor})"
                else:
                    performance_info = f"表現正常 (價值: {actual} vs 理論: {theor})"
            else:
                performance_info = f"狀態: 被牽制 (Pinned)"

            # 2. 機動性評估
            mobility_info = ""
            if actual_mobility == 0:
                mobility_info = "無法移動"
            elif theor_mobility > 0:
                mobility_ratio = actual_mobility / theor_mobility
                if mobility_ratio < 0.5:
                    mobility_info = f"機動性嚴重受限 ({actual_mobility}格 vs 理論{theor_mobility:.1f}格)"
                elif mobility_ratio < 0.8:
                    mobility_info = f"機動性受限 ({actual_mobility}格 vs 理論{theor_mobility:.1f}格)"
                elif mobility_ratio > 1.2:
                    mobility_info = f"機動性超常 ({actual_mobility}格 vs 理論{theor_mobility:.1f}格)"
                else:
                    mobility_info = f"機動性正常 ({actual_mobility}格 vs 理論{theor_mobility:.1f}格)"
            else:
                if actual_mobility <= 2:
                    mobility_info = f"機動性低 ({actual_mobility}格)"
                elif actual_mobility >= 6:
                    mobility_info = f"機動性高 ({actual_mobility}格)"
                else:
                    mobility_info = f"機動性中等 ({actual_mobility}格)"

            # 3. 戰術和策略洞察
            tactical_insights = []
            for pattern in patterns:
                tactical_insights.append(f"【{pattern['description']}】{pattern['chess_meaning']}")

            insights_str = " | ".join(tactical_insights) if tactical_insights else "常規表現"

            analysis_str += f"  - {p['piece']} ({p['square']}): {performance_info}, {mobility_info}\n"
            analysis_str += f"    戰術洞察: {insights_str}\n"

        return analysis_str

    white_piece_analysis_str = format_piece_analysis_enhanced(analysis_data['piece_values'], 'white')
    black_piece_analysis_str = format_piece_analysis_enhanced(analysis_data['piece_values'], 'black')
    
    # --- KMAPS 數據格式化 ---
    kmap_scores = analysis_data.get('kmap_scores', {})
    
    prompt = f"""
你是一位言簡意賅、極具洞察力的西洋棋主播。請根據以下資料，用**繁體中文**給出簡短且易懂的局面評論。注意：**禁止**在輸出中提到「K-MAPS、價值、理論值、機動性數值」等prompt數據，務必用自然語言描述。

－－－ 基本資訊 －－－
• FEN：`{analysis_data['fen']}`
• 評估結果（Stockfish）：{eval_term}（{eval_score/100:.2f}）
• 局面階段：{analysis_data['game_phase'].capitalize()}

以下是白方視角的結構評分：
• 王翼安全：{kmap_scores.get('K', 0):.2f}
• 材料優勢：{kmap_scores.get('M', 0):.2f}
• 子力活躍度：{kmap_scores.get('A', 0):.2f}
• 兵型結構：{kmap_scores.get('P', 0):.2f}
• 空間控制：{kmap_scores.get('S', 0):.2f}

－－－ 白方棋子概況（包含戰術洞察）－－－
{white_piece_analysis_str}

－－－ 黑方棋子概況（包含戰術洞察）－－－
{black_piece_analysis_str}

－－－ 高級戰術和策略判讀指南 －－－
根據以上數據中的「戰術洞察」部分，請特別注意：

1. **兵的表現判讀**：
   - 價值很高但機動受限 → 可能控制關鍵格、製造對手落後兵
   - 價值很高且機動性強 → 可能形成通路兵威脅、王前攻擊楔子
   - 價值異常低 → 可能導致王翼敞開、反向阻擋己方攻擊

2. **子力價值高時的判讀**：
   - 可能涉及將殺威脅、長將和棋
   - 可能準備發動戰術打擊（叉擊、雙重攻擊）
   - 可能控制關鍵通道或形成砲台

3. **機動性與價值不匹配**：
   - 高價值低機動 → 可能是戰術支撐點、重要防守子力
   - 低價值高機動 → 可能處於危險位置、過度進攻

4. **位置戰術意義**：
   - 中央控制、王翼安全、後排威脅
   - 被困子力、活躍子力的戰術含義

－－－ 任務說明 －－－
請**只**輸出以下 JSON 物件，字段不可增減。每一條評論 ≤ 25 字，務必結合戰術洞察使用自然語言。

{{
"overall_assessment": "一句話總結局面，結合 KMAPS 分數和戰術洞察。",
"white_positive_points": ["白方優點1", "白方優點2",...],
"white_negative_points": ["白方缺點1", "白方缺點2",...],
"black_positive_points": ["黑方優點1", "黑方優點2",...],
"black_negative_points": ["黑方缺點1", "黑方缺點2",...]
}}

－－－ 戰術風格範例 －－－
• "d5的馬控制了多個關鍵格子"
• "h6兵造成王翼結構敞開"
• "a1車被困在後排無法參與攻擊"
• "f7格成為致命弱點"
• "中央兵型形成強力楔子"
• "雙象控制了長對角線"
• "后翼兵型完全癱瘓"
• "e4兵阻擋了黑方主教發展"
"""
    print(prompt)
    print("--- Sending ENHANCED data (with tactical insights) to Google AI ---")
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        # 清理可能存在的 Markdown 標記
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        output_dict = json.loads(cleaned_text)
        return ChessAnalysis(**output_dict)
    except Exception as e:
        print(f"Error with LLM API or JSON parsing: {e}")
        # 打印原始回覆以進行調試
        print(f"Original response text: {response.text if 'response' in locals() else 'No response'}")
        return None
