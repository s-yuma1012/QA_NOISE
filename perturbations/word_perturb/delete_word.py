import random
from typing import Dict, List, Union
from fugashi import Tagger 
from datasets import Dataset
from tqdm import tqdm

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class DeleteWord:

    def __init__(self, data: Dataset, data_field: str='question', max_words: int=1, pos_tag: str=None):
        # 設定: 単語レベルでは max_perturbs (文字の繰り返し) は不要
        self.max_words: int = max_words
        self.pos_tag = pos_tag # 日本語品詞タグ (例: '名詞', '動詞')
        
        # データ設定
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'delete_word_jp'

    def execute_deletion(self, word_list: List[str]) -> List[str]:
        new_word_list = word_list.copy()
        
        if len(new_word_list) <= 1:
            return new_word_list
        
        # 削除する単語のインデックスをランダムに選択
        idx_to_delete = random.randrange(len(new_word_list))
        
        # 単語リストから削除を実行
        new_word_list.pop(idx_to_delete)
        
        return new_word_list

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]
        
        # 1. Fugashi (MeCab) を使用して単語と品詞に分割
        tokens = list(tagger(raw_text)) 
        
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
                print(f"-> Deleting Word: '{word_list[idx]}'")
                word_list.pop(idx)
                
            # 3-3. 単語リストを再結合 (スペース区切りは日本語では不自然だが、単語の境界を区別するために一時的に使用)
            perturbed_text = "".join(word_list) 
            
            # 4. 結果のキーに追加 (DWR: Delete Word Random)
            sample[f'{self.data_field}_perturbed_DWR'] = perturbed_text
            
        # 5. DEBUG表示
        print(f"Original Text: {raw_text}")
        print(f"Final Perturbed Text: {perturbed_text}")
        
        return sample

if __name__ == "__main__":
    from datasets import Dataset 
    
    test_sentences = [
        '9月1日の党代表選で選ばれた保守系の人物は？',
        '南アメリカの大国で、人口も多く、活気あふれる国として知られるところは。',
        '元は「日本共産党打倒」を掲げていた勢力が共産党と共に集会をする機会が増え始めたのはいつ以降？',
        '文春文庫はどこが出しているレーベル',
        '政府の経済政策による新工業化にもっとも寄与したのは何社？',
    ]
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': [''], 'context': ['']})
    
    # 例: DeleteWord
    attacker = DeleteWord(data=DUMMY_DATA, data_field='question', max_words=1, pos_tag=None)
    
    print(f"\n=== Word Level Perturbation Test ===")
    for i, sent in enumerate(test_sentences):
        dummy_sample = DUMMY_DATA[0].copy()
        dummy_sample['question'] = sent
        attacker.apply_on_sample(dummy_sample)