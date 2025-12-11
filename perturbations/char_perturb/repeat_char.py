import random
import copy
import re  # 正規表現用
from typing import Dict
from fugashi import Tagger 
from datasets import Dataset

"""
References = https://github.com/shalakasatheesh/robustness_eval_german_qa/
Modified for Japanese: Repeats only Hiragana characters (Chattering noise).
"""

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class RepeatChar:
    """
    日本語版の文字繰り返し攻撃クラス。
    【修正版】単語内の「ひらがな」のみをランダムに繰り返します。
    これはキーのチャタリング（二重入力）や長押しミスをシミュレートします。
    例: "しました" -> "しまました", "食べ物" -> "食べべ物" (漢字は繰り返さない)
    """
    def __init__(self, data: Dataset, data_field: str='question', max_perturbs: int=1, max_words: int=1, length_of_word_to_perturb: int=1, pos_tag: str=None):
        self.max_perturbs: int = max_perturbs
        self.max_words: int = max_words
        self.length_of_word_to_perturb = length_of_word_to_perturb
        self.pos_tag = pos_tag 
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'repeat_char_jp'

    def execute_repetition(self, word: str) -> str:
        """
        単語内のランダムな【ひらがな】を1文字複製するコアロジック
        """
        new_word: str = word
        
        for _ in range(self.max_perturbs):
            length = len(new_word)
            
            if length >= self.length_of_word_to_perturb:
                # ひらがなのインデックスのみを抽出
                hiragana_indices = [i for i, char in enumerate(new_word) if re.match(r'[\u3040-\u309F]', char)]
                
                if not hiragana_indices:
                    break # 繰り返せるひらがなが無い場合は終了
                
                # ひらがなの中からランダムに位置を選択
                idx_to_repeat = random.choice(hiragana_indices)
                
                # 複製した文字
                char_to_repeat = new_word[idx_to_repeat]
                
                # 文字列のスライスにより複製した文字を挿入
                new_word = new_word[:idx_to_repeat] + char_to_repeat + new_word[idx_to_repeat:]
            else:
                break 
        
        return new_word

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]

        # DEBUG: 攻撃前の文全体を表示
        print(f"\n[DEBUG-SENTENCE] --- START Attack on ID: {sample['id']} (Repetition) ---")
        print(f"[DEBUG-SENTENCE] Original Text: {raw_text}")
        
        tokens = list(tagger(raw_text)) 
        target_tokens = []
        
        for token in tokens:
            word_text = token.surface
            
            # ロバストな品詞取得
            try:
                feature_str = str(token.feature)
                if "pos1='" in feature_str:
                    start = feature_str.find("pos1='") + len("pos1='")
                    end = feature_str.find("'", start)
                    pos = feature_str[start:end]
                else:
                    pos = feature_str.split(',')[0]
            except:
                continue 
            
            if pos in ['助詞', '助動詞', '記号', 'BOS/EOS', '空白']:
                continue
            
            if (self.pos_tag is None or pos == self.pos_tag) and \
               len(word_text) >= self.length_of_word_to_perturb:
                
                # --- 修正箇所: ひらがなを含む単語のみ対象にする ---
                if re.search(r'[\u3040-\u309F]', word_text):
                    target_tokens.append(word_text)
        
        # 3. max_words の数だけランダムに単語を選択
        if not target_tokens:
            words_to_perturb = []
            print(f"[DEBUG-SENTENCE] -> Word selection skipped (No hiragana-containing words).")
        else:
            words_to_perturb = random.sample(target_tokens, min(self.max_words, len(target_tokens)))

        print(f"[DEBUG-SENTENCE] -> {len(words_to_perturb)} word(s) selected: {words_to_perturb}")
        
        perturbed_text = raw_text
        
        for original_word in words_to_perturb:
            print(f"[DEBUG-SENTENCE] -> Word Selected for Repetition: '{original_word}'")
            new_word = self.execute_repetition(original_word)
            
            # Simple replacement: 1回だけ置換
            perturbed_text = perturbed_text.replace(original_word, new_word, 1)

        sample[f'{self.data_field}_perturbed_RCR'] = perturbed_text

        print(f"[DEBUG-SENTENCE] Final Perturbed Text: {perturbed_text}")
        print("[DEBUG-SENTENCE] ----------------------------------")
        
        return sample

if __name__ == "__main__":
    # クイックテスト
    from datasets import Dataset 
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': ['ダミー質問'], 'context': ['ダミー文脈']})
    
    # テスト文:
    # 1. "東京大学" -> 漢字のみ (スキップされるべき)
    # 2. "しました" -> ひらがなのみ (繰り返されるべき)
    # 3. "食べ物" -> 漢字かな混じり (「べ」だけが候補になるべき)
    test_sentences = [
        "東京大学の研究。", 
        "確認しました。", 
        "美味しい食べ物。"
    ]

    # 攻撃インスタンスを作成
    repeater = RepeatChar(data=DUMMY_DATA, data_field='question', max_perturbs=1, length_of_word_to_perturb=1)

    print(f"\n--- [Quick Test: Hiragana Repetition] ---")
    
    for sent in test_sentences:
        dummy_sample = DUMMY_DATA[0].copy()
        dummy_sample['question'] = sent
        repeater.apply_on_sample(dummy_sample)