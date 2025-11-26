import random
import copy
from typing import Dict
from fugashi import Tagger 
from datasets import Dataset

"""
References = https://github.com/shalakasatheesh/robustness_eval_german_qa/

"""

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class InsertChar:
    """
    日本語版の文字挿入攻撃クラス。
    ランダムなひらがなを、単語内のランダムな位置に挿入します。
    """
    def __init__(self, data: Dataset, data_field: str='question', max_perturbs: int=1, max_words: int=1, length_of_word_to_perturb: int=2, pos_tag: str=None):
        # 攻撃の設定 (先行研究のパラメータを保持)
        self.max_perturbs: int = max_perturbs
        self.max_words: int = max_words
        self.length_of_word_to_perturb = length_of_word_to_perturb
        self.pos_tag = pos_tag # 日本語品詞タグ (例: '名詞', '動詞')
        
        # データ設定
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'insert_char_jp'

    def execute_insertion(self, word: str) -> str:
        """
        特定の単語内のランダムな位置に、ランダムなひらがなを挿入するコアロジック
        """
        new_word: str = word
        
        # --- 日本語化: 挿入する文字のプール ---
        # あ〜んまでのひらがな50音を定義
        HIRAGANA_POOL = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん" 

        for _ in range(self.max_perturbs):
            length = len(new_word)
            
            # length_of_word_to_perturb は削除ロジックの流用ですが、挿入では length >= length_of_word_to_perturb で許可
            if length >= self.length_of_word_to_perturb:
                
                # 挿入位置をランダムに選択 (0 から length まで。両端への挿入も許可)
                idx_to_insert = random.randrange(length + 1)
                
                # ランダムなひらがなを選択
                chosen_chara = random.choice(HIRAGANA_POOL)
                
                # 文字列のスライスにより挿入を実行
                new_word = new_word[:idx_to_insert] + chosen_chara + new_word[idx_to_insert:]
            else:
                break
        
        return new_word

    def apply_on_sample(self, sample: Dict) -> Dict:
        """
        datasets.map() 関数によって、個別のデータサンプルに攻撃を適用するラッパー
        """
        raw_text = sample[self.data_field]

        # DEBUG: 攻撃前の文全体を表示
        print(f"\n[DEBUG-SENTENCE] --- START Attack on ID: {sample['id']} (Insertion) ---")
        print(f"[DEBUG-SENTENCE] Target Field: {self.data_field}")
        print(f"[DEBUG-SENTENCE] Original Text: {raw_text}")
        
        # 1. Fugashi (MeCab) を使用して単語に分割
        tokens = list(tagger(raw_text)) 
        
        # 2. 攻撃対象の単語を選定 (POSタグと長さのフィルター)
        target_tokens = []
        for token in tokens:
            word_text = token.surface
            
            try:
                features = str(token.feature).split(',')
                pos = features[0] 
            except:
                # 品詞情報が取れない場合はスキップ
                continue 
            
            # 必須品詞のチェック (助詞/助動詞/記号/BOS/EOSは除外)
            if pos in ['助詞', '助動詞', '記号', 'BOS/EOS', '空白']:
                continue
            
            # POSタグ指定があるか、および長さのチェック
            if (self.pos_tag is None or pos == self.pos_tag) and \
               len(word_text) >= self.length_of_word_to_perturb: # 削除と異なり >= を使用
                target_tokens.append(word_text)
        
        # 3. max_words の数だけランダムに単語を選択
        words_to_perturb = random.sample(target_tokens, min(self.max_words, len(target_tokens)))

        # DEBAG: 攻撃対象として実際に選ばれた単語を表示
        print(f"[DEBUG-SENTENCE] -> {len(words_to_perturb)} word(s) selected: {words_to_perturb}")
        
        perturbed_text = raw_text
        
        # 4. 攻撃を実行し、置換
        for original_word in words_to_perturb:

            # DEBUG: どの単語が攻撃対象になったかを表示
            print(f"[DEBUG-SENTENCE] -> Word Selected for Insertion: '{original_word}'")

            # 挿入ロジックを実行
            new_word = self.execute_insertion(original_word)
            
            # Simple replacement: 1回だけ置換
            perturbed_text = perturbed_text.replace(original_word, new_word, 1)

        # 5. 結果を新しいキーに保存 (DCRと区別するため ICR を使用)
        sample[f'{self.data_field}_perturbed_ICR'] = perturbed_text

        # DEBUG: 攻撃後の文全体を表示
        print(f"[DEBUG-SENTENCE] Final Perturbed Text: {perturbed_text}")
        print("[DEBUG-SENTENCE] ----------------------------------")
        
        return sample

if __name__ == "__main__":
    # --- 1. テスト用のダミーデータ準備 ---
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': [''], 'context': ['']})
    
    # テスト対象の単語
    test_word = "コンピューター" # 7文字
    test_sentence = "今日、東京大学で重要な研究結果が発表された。"

    # 攻撃インスタンスを作成 (最低文字長を2に設定)
    deleter = InsertChar(data=DUMMY_DATA, data_field='question', max_perturbs=1, length_of_word_to_perturb=2)

    print(f"\n--- [Quick Test: execute_insertion (コアロジック)] ---")
    print(f"Original Word: {test_word} (Length: {len(test_word)})")
    
    # 挿入ロジックを複数回実行してランダム性を確認
    for i in range(3):
        result = deleter.execute_insertion(test_word)
        print(f"Test {i+1} Result: {result} (Length: {len(result)})")
    
    
    # --- 2. apply_on_sample のテスト ---
    dummy_sample = DUMMY_DATA[0].copy()
    dummy_sample['question'] = test_sentence
    
    print("\n--- [Quick Test: apply_on_sample (全体適用)] ---")
    
    # 攻撃インスタンスを作成 (最低文字長を1に設定し、確実に攻撃対象を選ぶ)
    inserter = InsertChar(data=DUMMY_DATA, data_field='question', max_perturbs=1, length_of_word_to_perturb=1)
    
    # apply_on_sampleを呼び出すと、内部でDEBUG printが出力される
    perturbed_sample = inserter.apply_on_sample(dummy_sample)
    
    print(f"Final Perturbed Sentence: {perturbed_sample['question_perturbed_ICR']}")
    print("----------------------------------")