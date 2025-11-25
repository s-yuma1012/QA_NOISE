import random
from typing import Dict, List, Union
from fugashi import Tagger 
from datasets import Dataset
from tqdm import tqdm

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class DeleteWord:
    """
    日本語版の単語削除攻撃クラス (DRQ, DRC, DVQ, DVCに対応)。
    ランダムな単語（または品詞指定の単語）を文脈から削除します。
    """
    def __init__(self, data: Dataset, data_field: str='question', max_words: int=1, pos_tag: str=None):
        # 設定: 単語レベルでは max_perturbs (文字の繰り返し) は不要
        self.max_words: int = max_words
        self.pos_tag = pos_tag # 日本語品詞タグ (例: '名詞', '動詞')
        
        # データ設定
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'delete_word_jp'

    def execute_deletion(self, word_list: List[str]) -> List[str]:
        """
        単語リストからランダムに単語を一つ削除するコアロジック
        """
        new_word_list = word_list.copy()
        
        if len(new_word_list) <= 1:
            return new_word_list
        
        # 削除する単語のインデックスをランダムに選択
        idx_to_delete = random.randrange(len(new_word_list))
        
        # 単語リストから削除を実行
        new_word_list.pop(idx_to_delete)
        
        return new_word_list

    def apply_on_sample(self, sample: Dict) -> Dict:
        """
        datasets.map() 関数によって、個別のデータサンプルに攻撃を適用するラッパー
        """
        raw_text = sample[self.data_field]
        
        # 1. Fugashi (MeCab) を使用して単語と品詞に分割
        tokens = list(tagger(raw_text)) 
        
        #print(f"[DIAGNOSTIC] All POS Tags: {[str(t.feature).split(',')[0] for t in tokens]}")
        # 2. 攻撃対象の単語を選定 (POSタグと長さのフィルター)
        # 今回は単語全体を削除するため、文字長チェックは省略可能だが、ここでは品詞フィルターに集中
        target_indices = [] # 攻撃対象となる単語のインデックスを格納
        
        for i, token in enumerate(tokens):
            word_text = token.surface
            try:
                features = str(token.feature)
                # 2. 'pos1=' の後にあるタグ名（例: '名詞'）を抽出
                if "pos1='" in features:
                    # 正規表現を使わず、文字列操作で抽出 (コードがシンプルになるため推奨)
                    start_index = features.find("pos1='") + len("pos1='")
                    end_index = features.find("'", start_index)
                    pos = features[start_index:end_index]
                else:
                # pos1が見つからない場合は、デフォルトで features[0] を使用 (保険)
                    pos = str(token.feature).split(',')[0]
                
            except:
                continue 

            # 助詞/助動詞/記号/空白の単語は攻撃対象から除外 (文の構造を大きく変えるため)
            if pos in ['助詞', '助動詞', '記号', 'BOS/EOS', '空白']:
                continue
            
            # POSタグ指定があるかどうかのチェック
            if self.pos_tag is None or pos == self.pos_tag:
                target_indices.append(i) # 攻撃可能な単語のインデックスを記録

        # 3. 攻撃単語の選定と実行
        if not target_indices:
            perturbed_text = raw_text
            sample[f'{self.data_field}_perturbed_DWR'] = raw_text
        else:
            # 攻撃する単語のインデックスをランダムに選択 (max_wordsの数だけ)
            indices_to_perturb = random.sample(target_indices, min(self.max_words, len(target_indices)))
            
            # 3-1. 攻撃を適用する前に、単語リストを再構築
            word_list = [t.surface for t in tokens]
            
            # 3-2. 攻撃対象のインデックスから単語を削除
            # インデックスの降順に削除することで、インデックスのズレを防ぐ
            indices_to_perturb.sort(reverse=True)
            for idx in indices_to_perturb:
                # DEBUG: 削除単語の表示
                print(f"[DEBUG-SENTENCE] -> Deleting Word: '{word_list[idx]}'")
                word_list.pop(idx)
                
            # 3-3. 単語リストを再結合 (スペース区切りは日本語では不自然だが、単語の境界を区別するために一時的に使用)
            perturbed_text = "".join(word_list) 
            
            # 4. 結果のキーに追加 (DWR: Delete Word Random)
            sample[f'{self.data_field}_perturbed_DWR'] = perturbed_text
            
        # 5. DEBUG表示
        print(f"\n[DEBUG-SENTENCE] Original Text: {raw_text}")
        print(f"[DEBUG-SENTENCE] Final Perturbed Text: {perturbed_text}")
        print("----------------------------------")
        
        return sample

if __name__ == "__main__":
    # --- 1. テスト用のダミーデータ準備 ---
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': ['ダミー質問'], 'context': ['ダミー文脈']})
    
    # 助詞・記号を含むテスト文
    test_sentence_random = "日本の首相が、新しい研究開発の予算を決定した。"
    test_sentence_pos = "東京大学の研究結果が、広く知られています。"
    
    # 攻撃インスタンスを作成
    # 1. POSタグ指定なし（ランダムな名詞/動詞などを削除）
    deleter_random = DeleteWord(data=DUMMY_DATA, data_field='question', max_words=1, pos_tag=None) 
    # 2. POSタグ指定あり（名詞 '名詞' のみを削除）
    deleter_noun = DeleteWord(data=DUMMY_DATA, data_field='question', max_words=1, pos_tag='名詞')

    print(f"\n--- [Quick Test: DeleteWord - 汎用攻撃] ---")
    
    # 1. 汎用攻撃のテスト
    dummy_sample_random = DUMMY_DATA[0].copy()
    dummy_sample_random['question'] = test_sentence_random
    print(f"Original Text (Random): {test_sentence_random}")
    
    for i in range(3):
        # 連続実行してランダム性とロジック崩壊がないかを確認
        perturbed_sample = deleter_random.apply_on_sample(dummy_sample_random.copy())
        print(f"Test {i+1} Result: {perturbed_sample['question_perturbed_DWR']}")


    print(f"\n--- [Quick Test: DeleteWord - 名詞限定攻撃] ---")
    
    # 2. 名詞限定攻撃のテスト
    dummy_sample_pos = DUMMY_DATA[0].copy()
    dummy_sample_pos['question'] = test_sentence_pos
    print(f"Original Text (Noun Only): {test_sentence_pos}")
    
    # apply_on_sampleを呼び出す
    perturbed_sample = deleter_noun.apply_on_sample(dummy_sample_pos)
    print(f"Final Perturbed Sentence: {perturbed_sample['question_perturbed_DWR']}")
    print("----------------------------------")