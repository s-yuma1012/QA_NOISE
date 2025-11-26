import random
from typing import Dict, List, Union
from fugashi import Tagger 
from datasets import Dataset
from tqdm import tqdm

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class SwapWord:
    """
    日本語版の単語交換攻撃クラス (SwapRQ, SwapRCなどに対応)。
    文中のランダムな2つの単語（または品詞指定の単語）の位置を入れ替えます。
    例: "私が学校に行く" -> "学校が私に行く"
    """
    def __init__(self, data: Dataset, data_field: str='question', max_words: int=1, pos_tag: str=None):
        # max_words は「何組のペアを交換するか」として解釈します
        self.max_words: int = max_words 
        self.pos_tag = pos_tag # 日本語品詞タグ (例: '名詞', '動詞')
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'swap_word_jp'

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]
        
        # DEBUG: 処理開始と元のテキストを表示
        print(f"\n[DEBUG-SENTENCE] Original Text: {raw_text}")

        tokens = list(tagger(raw_text)) 
        target_indices = [] 
        
        # 1. 攻撃対象（交換候補）となる単語のインデックスを収集
        for i, token in enumerate(tokens):
            # --- 品詞タグ取得 ---
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
        # 交換には最低2つの単語が必要
        if len(target_indices) < 2:
            perturbed_text = raw_text
            print(f"[DEBUG-SENTENCE] -> Word selection skipped (Need at least 2 eligible words for swap).")
        else:
            # 単語リストを作成
            word_list = [t.surface for t in tokens]
            
            # 交換操作を max_words 回繰り返す
            # (注: ここでの max_words は「交換ペアの数」を意味します)
            swap_count = 0
            
            # 安全のため、リストをコピーして操作する
            target_indices_pool = target_indices.copy()
            
            while swap_count < self.max_words and len(target_indices_pool) >= 2:
                # ランダムに2つのインデックスを選択
                pair = random.sample(target_indices_pool, 2)
                idx1, idx2 = pair[0], pair[1]
                
                # DEBUG: 交換する単語を表示
                print(f"[DEBUG-SENTENCE] -> Swapping Words: '{word_list[idx1]}' <-> '{word_list[idx2]}'")
                
                # リスト上で直接スワップ
                word_list[idx1], word_list[idx2] = word_list[idx2], word_list[idx1]
                
                # 同じ単語を何度も交換しないよう、プールから除外する（オプション）
                # 今回はシンプルにするため、除外せずに独立試行としても良いが、
                # ロジックを明確にするため、一度使ったインデックスはプールから削除する
                target_indices_pool.remove(idx1)
                target_indices_pool.remove(idx2)
                
                swap_count += 1
                
            perturbed_text = "".join(word_list)
            
        # 結果を格納 (SWR: Swap Word Random)
        sample[f'{self.data_field}_perturbed_SWR'] = perturbed_text
        
        # DEBUG: 最終結果を表示
        print(f"[DEBUG-SENTENCE] Final Perturbed Text: {perturbed_text}")
        print("----------------------------------")
        
        return sample

if __name__ == "__main__":
    # クイックテスト
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': [''], 'context': ['']})
    
    # テスト文
    test_sentence = "東京大学の研究結果が、広く知られています。"
    
    # 1. 名詞同士を交換
    swapper = SwapWord(data=DUMMY_DATA, data_field='question', max_words=1, pos_tag='名詞')
    
    dummy_sample = DUMMY_DATA[0].copy()
    dummy_sample['question'] = test_sentence
    
    result = swapper.apply_on_sample(dummy_sample)