import random
import copy
import re  
from typing import Dict
from fugashi import Tagger 
from datasets import Dataset

"""
References = https://github.com/shalakasatheesh/robustness_eval_german_qa/

"""

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class DeleteChar:
    """
    日本語版の文字削除攻撃クラス。
    【修正版】ひらがなのみを削除対象とします。
    漢字やカタカナの削除は入力ミスとして不自然なため除外します。
    """
    def __init__(self, data: Dataset, data_field: str='question', max_perturbs: int=1, max_words: int=1, length_of_word_to_perturb: int=2, pos_tag: str=None):
        self.max_perturbs: int = max_perturbs
        self.max_words: int = max_words
        self.length_of_word_to_perturb = length_of_word_to_perturb
        self.pos_tag = pos_tag 
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'delete_char_jp'

    def execute_deletion(self, word: str) -> str:
        """
        特定の単語からランダムに1文字（ただしひらがな限定）を削除するコアロジック
        """
        new_word: str = word
        
        for _ in range(self.max_perturbs):
            length = len(new_word)
            
            # 文字数チェック
            if length > self.length_of_word_to_perturb:
                # --- 修正箇所: ひらがなのインデックスのみを候補にする ---
                # Unicode範囲: \u3040-\u309F がひらがな
                hiragana_indices = [i for i, char in enumerate(new_word) if re.match(r'[\u3040-\u309F]', char)]
                
                if not hiragana_indices:
                    break # 削除できるひらがなが無い場合は終了
                
                # ひらがなの中からランダムに削除位置を選択
                idx_to_delete = random.choice(hiragana_indices)
                
                new_word = new_word[:idx_to_delete] + new_word[idx_to_delete + 1:]
            else:
                break
        
        return new_word

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]

        # DEBUG: 攻撃前の文全体を表示
        print(f"\n[DEBUG-SENTENCE] --- START Attack on ID: {sample['id']} ---")
        print(f"[DEBUG-SENTENCE] Original Text: {raw_text}")
        
        tokens = list(tagger(raw_text)) 
        
        target_tokens = []
        for token in tokens:
            word_text = token.surface
            
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
            
            # POSタグ指定があるか、および長さのチェック
            if (self.pos_tag is None or pos == self.pos_tag) and \
               len(word_text) > self.length_of_word_to_perturb:
                
                # --- 修正箇所: ひらがなを含まない単語は最初から除外する ---
                # (例: "東京大学" はひらがなが無いので対象外になる)
                if re.search(r'[\u3040-\u309F]', word_text):
                    target_tokens.append(word_text)
        
        # 3. max_words の数だけランダムに単語を選択
        if not target_tokens:
            words_to_perturb = []
            print("[DEBUG-SENTENCE] -> No suitable words found (No hiragana-containing words).")
        else:
            words_to_perturb = random.sample(target_tokens, min(self.max_words, len(target_tokens)))

        print(f"[DEBUG-SENTENCE] -> {len(words_to_perturb)} word(s) selected: {words_to_perturb}")
        
        perturbed_text = raw_text
        
        for original_word in words_to_perturb:
            print(f"[DEBUG-SENTENCE] -> Word Selected for Deletion: '{original_word}'")
            new_word = self.execute_deletion(original_word)
            
            # Simple replacement: 1回だけ置換
            perturbed_text = perturbed_text.replace(original_word, new_word, 1)

        sample[f'{self.data_field}_perturbed_DCR'] = perturbed_text

        print(f"[DEBUG-SENTENCE] Final Perturbed Text: {perturbed_text}")
        print("[DEBUG-SENTENCE] ----------------------------------")
        
        return sample

if __name__ == "__main__":
    # クイックテスト
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': ['ダミー'], 'context': ['ダミー']})
    
    test_sentences = [
        '9月1日の党代表選で選ばれた保守系の人物は？',
        '南アメリカの大国で、人口も多く、活気あふれる国として知られるところは。',
        '元は「日本共産党打倒」を掲げていた勢力が共産党と共に集会をする機会が増え始めたのはいつ以降？',
        '文春文庫はどこが出しているレーベル',
        '政府の経済政策による新工業化にもっとも寄与したのは何社？',
    ]

    # length_of_word_to_perturb=0 にして、短い単語もテスト対象にする
    deleter = DeleteChar(data=DUMMY_DATA, data_field='question', max_perturbs=1, length_of_word_to_perturb=0)

    print(f"\n--- [Quick Test: Hiragana Deletion] ---")
    
    for sent in test_sentences:
        dummy_sample = DUMMY_DATA[0].copy()
        dummy_sample['question'] = sent
        deleter.apply_on_sample(dummy_sample)