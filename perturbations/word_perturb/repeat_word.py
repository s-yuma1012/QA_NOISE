import random
from typing import Dict, List, Union
from fugashi import Tagger 
from datasets import Dataset
from tqdm import tqdm

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class RepeatWord:
    """
    日本語版の単語繰り返し攻撃クラス (RepeatWR, RepeatVRなどに対応)。
    ランダムな単語（または品詞指定の単語）を複製して直後に挿入します。
    例: "東京大学" -> "東京大学東京大学"
    """
    def __init__(self, data: Dataset, data_field: str='question', max_words: int=1, pos_tag: str=None):
        self.max_words: int = max_words
        self.pos_tag = pos_tag # 日本語品詞タグ (例: '名詞', '動詞')
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'repeat_word_jp'

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]
        
        # DEBUG: 処理開始と元のテキストを表示
        print(f"\n[DEBUG-SENTENCE] Original Text: {raw_text}")

        tokens = list(tagger(raw_text)) 
        target_indices = [] 
        
        for i, token in enumerate(tokens):
            # --- 品詞タグ取得 (delete_word.pyと同じロバストなロジック) ---
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
            
            if self.pos_tag is None or pos == self.pos_tag:
                target_indices.append(i)

        # --- 攻撃実行ロジック ---
        if not target_indices:
            perturbed_text = raw_text
            # DEBUG: スキップされた場合
            print(f"[DEBUG-SENTENCE] -> Word selection skipped (No eligible words found for POS '{self.pos_tag}').")
        else:
            # 単語リストを作成
            word_list = [t.surface for t in tokens]
            
            # 攻撃対象をランダムに選択
            indices_to_perturb = random.sample(target_indices, min(self.max_words, len(target_indices)))
            
            # 【重要】インデックスのズレを防ぐため、降順（後ろから）処理する
            indices_to_perturb.sort(reverse=True)
            
            for idx in indices_to_perturb:
                word_to_repeat = word_list[idx]
                
                # DEBUG: 繰り返す単語を表示
                print(f"[DEBUG-SENTENCE] -> Repeating Word: '{word_to_repeat}'")
                
                # リストの該当位置に同じ単語を挿入 (例: [A, B, C] -> idx=1(B) -> [A, B, B, C])
                word_list.insert(idx, word_to_repeat)
                
            perturbed_text = "".join(word_list)
            
        # 結果を格納 (RWR: Repeat Word Random)
        sample[f'{self.data_field}_perturbed_RWR'] = perturbed_text
        
        # DEBUG: 最終結果を表示
        print(f"[DEBUG-SENTENCE] Final Perturbed Text: {perturbed_text}")
        print("----------------------------------")
        
        return sample

if __name__ == "__main__":
    # クイックテスト
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': [''], 'context': ['']})
    
    # テスト文
    test_sentence = "日本の首相が、新しい研究開発の予算を決定した。"
    
    # 1. 名詞を繰り返す
    repeater = RepeatWord(data=DUMMY_DATA, data_field='question', max_words=1, pos_tag='名詞')
    
    dummy_sample = DUMMY_DATA[0].copy()
    dummy_sample['question'] = test_sentence
    
    # apply_on_sampleを実行 (内部でprintされる)
    result = repeater.apply_on_sample(dummy_sample)