import random
import copy
from typing import Dict, List, Union
from fugashi import Tagger 
from datasets import Dataset
from tqdm import tqdm

"""
References = https://github.com/shalakasatheesh/robustness_eval_german_qa/
Modified for Japanese: Replaces Particles (助詞) with other Particles.
"""

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class ReplaceChar:
    """
    日本語版の文字置換攻撃クラス。
    【修正版】「助詞」を、別の「助詞」に置換します。
    これは助詞の誤用（例：「ご飯を食べる」→「ご飯が食べる」）をシミュレートします。
    """
    def __init__(self, data: Dataset, data_field: str='question', max_perturbs: int=1, max_words: int=1, length_of_word_to_perturb: int=0, pos_tag: str=None):
        self.max_perturbs: int = max_perturbs
        self.max_words: int = max_words
        self.length_of_word_to_perturb = length_of_word_to_perturb
        self.pos_tag = pos_tag 
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'replace_char_jp'
        
    def execute_replacement(self, word: str) -> str:
        """
        助詞を別の助詞に置換するコアロジック
        """
        # 代表的な助詞のプール (格助詞、係助詞など)
        PARTICLE_POOL = ["が", "の", "を", "に", "へ", "と", "で", "や", "も", "は"]
        
        # 元の助詞と異なるもの候補リストを作成
        candidates = [p for p in PARTICLE_POOL if p != word]
        
        # もし元の単語がプールにない場合（例：「より」「から」など）、プール全体から選ぶ
        if len(candidates) == len(PARTICLE_POOL):
             # 元の単語がプールに無い場合でも、念のため元の単語と同じにならないようにチェック（確率低いですが）
             new_word = random.choice(PARTICLE_POOL)
        else:
             # 元の単語を除外したリストから選ぶ
             new_word = random.choice(candidates)
        
        return new_word

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]

        # DEBUG: 攻撃前の文全体を表示
        print(f"\n[DEBUG-SENTENCE] --- START Attack on ID: {sample['id']} (Particle Replacement) ---")
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

            # 助詞のみをターゲットにする
            if pos == '助詞':
                # 長さチェック (length_of_word_to_perturb=0なら全ての助詞を対象)
                if len(word_text) >= self.length_of_word_to_perturb:
                    target_tokens.append(word_text)
        
        # 3. 攻撃実行
        if not target_tokens:
            words_to_perturb = []
            print(f"[DEBUG-SENTENCE] -> Word selection skipped (No particles found).")
        else:
            words_to_perturb = random.sample(target_tokens, min(self.max_words, len(target_tokens)))

        print(f"[DEBUG-SENTENCE] -> {len(words_to_perturb)} particle(s) selected: {words_to_perturb}")
        
        perturbed_text = raw_text
        
        for original_word in words_to_perturb:
            new_word = self.execute_replacement(original_word)
            
            print(f"[DEBUG-SENTENCE] -> Particle Swap: '{original_word}' -> '{new_word}'")
            
            # Simple replacement: 1回だけ置換
            perturbed_text = perturbed_text.replace(original_word, new_word, 1)

        sample[f'{self.data_field}_perturbed_RCR'] = perturbed_text

        print(f"[DEBUG-SENTENCE] Final Perturbed Text: {perturbed_text}")
        print("[DEBUG-SENTENCE] ----------------------------------")
        
        return sample

if __name__ == "__main__":
    from datasets import Dataset 
    
    # JSQuADから抽出した実データ5件
    test_sentences = [
        '9月1日の党代表選で選ばれた保守系の人物は？',
        '南アメリカの大国で、人口も多く、活気あふれる国として知られるところは。',
        '元は「日本共産党打倒」を掲げていた勢力が共産党と共に集会をする機会が増え始めたのはいつ以降？',
        '文春文庫はどこが出しているレーベル',
        '政府の経済政策による新工業化にもっとも寄与したのは何社？',
    ]
    
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': [''], 'context': ['']})

    print(f"\n=== Test Runner with JSQuAD Samples ===")
    
    attacker = ReplaceChar(data=DUMMY_DATA, data_field='question', max_perturbs=1, length_of_word_to_perturb=0)

    for i, sent in enumerate(test_sentences):
        print(f"\n--- Sample {i+1} ---")
        dummy_sample = DUMMY_DATA[0].copy()
        dummy_sample['question'] = sent
        
        # 実行 (内部でDEBUGプリントが出ます)
        attacker.apply_on_sample(dummy_sample)