import random
import re
import jaconv
from typing import Dict
from fugashi import Tagger 
from datasets import Dataset

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class KatakanaToHiragana:
    
    def __init__(self, data: Dataset, data_field: str='question', max_words: int=1, pos_tag: str=None):
        self.max_words: int = max_words
        self.pos_tag = pos_tag 
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'katakana_to_hiragana_jp'

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]
        
        
        print(f"Original Text: {raw_text}")
        
        tokens = list(tagger(raw_text)) 
        target_indices = []
        
        for i, token in enumerate(tokens):
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
                    target_indices.append(i)
        
        # 攻撃実行
        if not target_indices:
            perturbed_text = raw_text
            print(f"-> Word selection skipped (No Katakana-containing words).")
        else:
            word_list = [t.surface for t in tokens]
            indices_to_perturb = random.sample(target_indices, min(self.max_words, len(target_indices)))
            
            for idx in indices_to_perturb:
                original_word = word_list[idx]
                # jaconvで変換
                new_word = jaconv.kata2hira(original_word)
                
                print(f"-> Converted: '{original_word}' -> '{new_word}'")
                word_list[idx] = new_word
            
            perturbed_text = "".join(word_list)

        sample[f'{self.data_field}_perturbed_K2H'] = perturbed_text
        print(f"Final Perturbed Text: {perturbed_text}")
        
        
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
    
    print(f"\n=== Hiragana to Katakana (1 Char) Test (JSQuAD) ===")
    
    converter = KatakanaToHiragana(data=DUMMY_DATA, data_field='question', max_words=3)
    
    for i, sent in enumerate(test_sentences):
        dummy_sample = DUMMY_DATA[0].copy()
        dummy_sample['question'] = sent
        converter.apply_on_sample(dummy_sample)