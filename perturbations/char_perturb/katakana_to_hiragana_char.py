import random
import re
import jaconv  # ひらがな⇔カタカナ変換用
from typing import Dict
from fugashi import Tagger 
from datasets import Dataset

"""
Katakana to Hiragana Conversion (Single Character)
Converts ONE Katakana character in a word to Hiragana.
Example: "コンピューター" -> "コンぴューター"
"""

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class KatakanaToHiragana:
    """
    日本語版の文字変換攻撃クラス。
    単語に含まれる「カタカナ」のうち【一文字だけ】を「ひらがな」に変換します。
    """
    def __init__(self, data: Dataset, data_field: str='question', max_words: int=1, pos_tag: str=None):
        self.max_words: int = max_words
        self.pos_tag = pos_tag 
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'katakana_to_hiragana_char_jp'

    def execute_conversion(self, word: str) -> str:
        """
        単語内のランダムなカタカナ1文字をひらがなに変換する
        """
        # カタカナのインデックスを全て取得 (\u30A0-\u30FF)
        katakana_indices = [i for i, char in enumerate(word) if re.match(r'[\u30A0-\u30FF]', char)]
        
        if not katakana_indices:
            return word
            
        # 変換する位置をランダムに1つ選択
        target_idx = random.choice(katakana_indices)
        
        # その1文字だけ変換
        target_char = word[target_idx]
        new_char = jaconv.kata2hira(target_char)
        
        # 文字列再構築
        new_word = word[:target_idx] + new_char + word[target_idx+1:]
        
        return new_word

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]

        # DEBUG
        print(f"\n[DEBUG-SENTENCE] --- START Attack on ID: {sample['id']} (Kata->Hira 1char) ---")
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
            
            if pos in ['记号', 'BOS/EOS', '空白']:
                continue
            
            if (self.pos_tag is None or pos == self.pos_tag):
                # カタカナを含む単語のみ対象
                if re.search(r'[\u30A0-\u30FF]', word_text):
                    target_tokens.append(word_text)
        
        if not target_tokens:
            words_to_perturb = []
            print(f"[DEBUG-SENTENCE] -> Word selection skipped.")
        else:
            words_to_perturb = random.sample(target_tokens, min(self.max_words, len(target_tokens)))

        print(f"[DEBUG-SENTENCE] -> {len(words_to_perturb)} word(s) selected: {words_to_perturb}")
        
        perturbed_text = raw_text
        
        for original_word in words_to_perturb:
            new_word = self.execute_conversion(original_word)
            
            print(f"[DEBUG-SENTENCE] -> Converted: '{original_word}' -> '{new_word}'")
            perturbed_text = perturbed_text.replace(original_word, new_word, 1)

        sample[f'{self.data_field}_perturbed_K2H'] = perturbed_text
        print(f"[DEBUG-SENTENCE] Final Perturbed Text: {perturbed_text}")
        print("[DEBUG-SENTENCE] ----------------------------------")
        
        return sample

if __name__ == "__main__":
    from datasets import Dataset 
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': ['ダミー'], 'context': ['ダミー']})
    test_sentences = ["新しいコンピューター。", "サービスエリア。"]
    converter = KatakanaToHiragana(data=DUMMY_DATA, data_field='question', max_words=1)
    
    print(f"\n--- [Quick Test: Katakana to Hiragana (1 Char)] ---")
    for sent in test_sentences:
        dummy_sample = DUMMY_DATA[0].copy()
        dummy_sample['question'] = sent
        converter.apply_on_sample(dummy_sample)