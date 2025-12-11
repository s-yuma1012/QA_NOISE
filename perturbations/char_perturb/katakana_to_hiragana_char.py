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
            
            if pos in ['記号', 'BOS/EOS', '空白']:
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
    
    # カタカナが含まれる文(2, 4番目)以外はスキップされる
    attacker = KatakanaToHiragana(data=DUMMY_DATA, data_field='question', max_words=2)
    
    for i, sent in enumerate(test_sentences):
        print(f"\n--- Sample {i+1} ---")
        dummy_sample = DUMMY_DATA[0].copy()
        dummy_sample['question'] = sent
        
        # 実行 (内部でDEBUGプリントが出ます)
        attacker.apply_on_sample(dummy_sample)