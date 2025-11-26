import random
import copy
from typing import Dict, List, Union
from fugashi import Tagger 
from datasets import Dataset
from tqdm import tqdm

"""
References = https://github.com/shalakasatheesh/robustness_eval_german_qa/
Character Replacement (RCRQ, RCRC, RCVQ, RCVC)
"""

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class ReplaceChar:
    """
    日本語版の文字置換攻撃クラス。
    単語内の既存の文字を、ランダムなひらがなに置き換えます。
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
        self.name: str = 'replace_char_jp'
        
        # カウンターと記録用コンテナの初期化 (省略)


    def execute_replacement(self, word: str) -> str:
        """
        特定の単語内のランダムな位置の文字を、ランダムなひらがなに置換するコアロジック
        """
        new_word: str = word
        
        # --- 日本語化: 置換する文字のプール (ひらがな50音) ---
        HIRAGANA_POOL = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん" 

        for _ in range(self.max_perturbs):
            length = len(new_word)
            
            # 最小文字長チェック: 指定された長さ以上の単語のみ実行
            if length > self.length_of_word_to_perturb:
                
                # 置換位置をランダムに選択 (0 から length - 1 まで)
                # 今回は端の文字も置換対象とします
                idx_to_replace = random.randrange(length)
                
                # ランダムなひらがなを選択
                chosen_chara = random.choice(HIRAGANA_POOL)
                
                # 文字列のスライスにより置換を実行
                # 例: "テスト" (idx=1, char='ん') -> "テ" + "ん" + "スト" = "テンスト"
                new_word = new_word[:idx_to_replace] + chosen_chara + new_word[idx_to_replace + 1:]
            else:
                break # 条件を満たさない場合はスキップ
        
        return new_word

    def apply_on_sample(self, sample: Dict) -> Dict:
        """
        datasets.map() 関数によって、個別のデータサンプルに攻撃を適用するラッパー
        (DeleteCharのロジックとほぼ同一)
        """
        raw_text = sample[self.data_field]

        # DEBUG: 攻撃前の文全体を表示
        print(f"\n[DEBUG-SENTENCE] --- START Attack on ID: {sample['id']} (Replacement) ---")
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
                continue 

            if pos in ['助詞', '助動詞', '記号', 'BOS/EOS', '空白']:
                continue
            
            if (self.pos_tag is None or pos == self.pos_tag) and \
               len(word_text) > self.length_of_word_to_perturb:
                target_tokens.append(word_text)
        
        # 3. max_words の数だけランダムに単語を選択
        words_to_perturb = random.sample(target_tokens, min(self.max_words, len(target_tokens)))

        # DEBAG: 攻撃対象として実際に選ばれた単語を表示
        print(f"[DEBUG-SENTENCE] -> {len(words_to_perturb)} word(s) selected: {words_to_perturb}")
        
        perturbed_text = raw_text
        
        # 4. 攻撃を実行し、置換
        for original_word in words_to_perturb:

            # DEBUG: どの単語が攻撃対象になったかを表示
            print(f"[DEBUG-SENTENCE] -> Word Selected for Replacement: '{original_word}'")

            # コアロジックを実行
            new_word = self.execute_replacement(original_word)
            
            # Simple replacement: 1回だけ置換
            perturbed_text = perturbed_text.replace(original_word, new_word, 1)

        # 5. 結果を新しいキーに保存 (RCR: Replace Character)
        sample[f'{self.data_field}_perturbed_RCR'] = perturbed_text

        # DEBUG: 攻撃後の文全体を表示
        print(f"[DEBUG-SENTENCE] Final Perturbed Text: {perturbed_text}")
        print("[DEBUG-SENTENCE] ----------------------------------")
        
        return sample

if __name__ == "__main__":
    # --- Quick Test: execute_replacement (コアロジック) ---
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': [''], 'context': ['']})
    
    test_word = "コンピューター"
    test_sentence = "今日、東京大学で重要な研究結果が発表された。"

    # 攻撃インスタンスを作成 (最低文字長を2に設定)
    deleter = ReplaceChar(data=DUMMY_DATA, data_field='question', max_perturbs=1, length_of_word_to_perturb=2)

    print(f"\n--- [Quick Test: execute_replacement (コアロジック)] ---")
    print(f"Original Word: {test_word} (Length: {len(test_word)})")
    
    for i in range(3):
        result = deleter.execute_replacement(test_word)
        print(f"Test {i+1} Result: {result} (Length: {len(result)})")
    
    
    # --- Quick Test: apply_on_sample (全体適用) ---
    dummy_sample = DUMMY_DATA[0].copy()
    dummy_sample['question'] = test_sentence
    
    print("\n--- [Quick Test: apply_on_sample (全体適用)] ---")
    
    # 攻撃インスタンスを作成 (最低文字長を1に設定)
    replacer = ReplaceChar(data=DUMMY_DATA, data_field='question', max_perturbs=1, length_of_word_to_perturb=1)
    
    # apply_on_sampleを呼び出す
    perturbed_sample = replacer.apply_on_sample(dummy_sample)
    
    print(f"Final Perturbed Sentence: {perturbed_sample['question_perturbed_RCR']}")
    print("----------------------------------")