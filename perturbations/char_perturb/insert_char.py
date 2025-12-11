import random
import copy
import string  # アルファベット取得用
from typing import Dict
from fugashi import Tagger 
from datasets import Dataset

"""
References = https://github.com/shalakasatheesh/robustness_eval_german_qa/
Modified for Japanese: Appends Alphabet characters to the end of the word.
"""

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class InsertChar:
    """
    日本語版の文字挿入攻撃クラス (ICRQ, ICRC, ICVQ, ICVCに対応)。
    【修正版】ランダムなアルファベット(a-z, A-Z)を単語の【末尾】に追加します。
    これは入力確定後のミスタッチや、次の文字の入力漏れなどをシミュレートします。
    例: "東京大学" -> "東京大学a"
    """
    def __init__(self, data: Dataset, data_field: str='question', max_perturbs: int=1, max_words: int=1, length_of_word_to_perturb: int=2, pos_tag: str=None):
        self.max_perturbs: int = max_perturbs
        self.max_words: int = max_words
        self.length_of_word_to_perturb = length_of_word_to_perturb
        self.pos_tag = pos_tag 
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'insert_char_jp'

    def execute_insertion(self, word: str) -> str:
        """
        特定の単語の【末尾】に、ランダムなアルファベットを挿入するコアロジック
        """
        new_word: str = word
        
        # アルファベットのプール
        ALPHABET_POOL = string.ascii_lowercase  # abc...xyz

        for _ in range(self.max_perturbs):
            # 文字数チェック
            length = len(new_word)
            
            if length >= self.length_of_word_to_perturb:
                # ランダムなアルファベットを選択
                chosen_chara = random.choice(ALPHABET_POOL)
                
                # --- 修正箇所: 常に単語の「後ろ」に追加する ---
                new_word = new_word + chosen_chara
            else:
                break
        
        return new_word

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]

        # DEBUG: 攻撃前の文全体を表示
        print(f"\n[DEBUG-SENTENCE] --- START Attack on ID: {sample['id']} (Insertion) ---")
        print(f"[DEBUG-SENTENCE] Original Text: {raw_text}")
        
        tokens = list(tagger(raw_text)) 
        target_tokens = []
        
        for token in tokens:
            word_text = token.surface
            
            # ロバストな品詞取得ロジック
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
            
            # 必須品詞のチェック (助詞/助動詞/記号/BOS/EOSは除外)
            if pos in ['助詞', '助動詞', '記号', 'BOS/EOS', '空白']:
                continue
            
            # POSタグ指定があるか、および長さのチェック
            if (self.pos_tag is None or pos == self.pos_tag) and \
               len(word_text) >= self.length_of_word_to_perturb:
                target_tokens.append(word_text)
        
        # 3. max_words の数だけランダムに単語を選択
        if not target_tokens:
            words_to_perturb = []
            print(f"[DEBUG-SENTENCE] -> Word selection skipped (No eligible words found).")
        else:
            words_to_perturb = random.sample(target_tokens, min(self.max_words, len(target_tokens)))

        print(f"[DEBUG-SENTENCE] -> {len(words_to_perturb)} word(s) selected: {words_to_perturb}")
        
        perturbed_text = raw_text
        
        for original_word in words_to_perturb:
            print(f"[DEBUG-SENTENCE] -> Word Selected for Insertion: '{original_word}'")
            new_word = self.execute_insertion(original_word)
            
            # Simple replacement: 1回だけ置換
            perturbed_text = perturbed_text.replace(original_word, new_word, 1)

        sample[f'{self.data_field}_perturbed_ICR'] = perturbed_text

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
    
    
    attacker = InsertChar(data=DUMMY_DATA, data_field='question', max_perturbs=1, length_of_word_to_perturb=1)
    
    for i, sent in enumerate(test_sentences):
        print(f"\n--- Sample {i+1} ---")
        dummy_sample = DUMMY_DATA[0].copy()
        dummy_sample['question'] = sent
        
        # 実行 (内部でDEBUGプリントが出ます)
        attacker.apply_on_sample(dummy_sample)