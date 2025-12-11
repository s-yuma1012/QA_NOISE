import random
from typing import Dict, List
from fugashi import Tagger
from datasets import Dataset
from transformers import pipeline
from tqdm import tqdm

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class SynonymReplace:
    """
    日本語版の同義語置換攻撃クラス (Synonym Replacement)。
    MLM (RoBERTa) を使用して、文脈を考慮しながら単語を別の表現に置き換えます。
    例: "彼は速く走る" -> "彼は速く動く"
    """
    def __init__(self, data: Dataset, data_field: str='question', max_words: int=1, pos_tag: str=None, model_name: str="tohoku-nlp/bert-base-japanese-whole-word-masking"):
        self.max_words: int = max_words
        self.pos_tag = pos_tag # 日本語品詞タグ (例: '動詞', '形容詞')
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'synonym_replace_jp'
        
        # MLMパイプラインの初期化 (ここが少し重いです)
        print(f"[INFO] Loading MLM pipeline ({model_name})...")
        self.fill_mask = pipeline("fill-mask", model=model_name, top_k=5)
        self.mask_token = self.fill_mask.tokenizer.mask_token # モデルに合わせて [MASK] などを取得

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]
        
        # DEBUG: 処理開始
        print(f"\n[DEBUG-SENTENCE] Original Text: {raw_text}")

        # 1. 単語分割
        tokens = list(tagger(raw_text)) 
        target_indices = [] 
        
        # 2. 攻撃対象の選定 (ロバストな品詞取得)
        for i, token in enumerate(tokens):
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

        # 3. 攻撃実行ロジック
        if not target_indices:
            perturbed_text = raw_text
            print(f"[DEBUG-SENTENCE] -> Word selection skipped (No eligible words for POS '{self.pos_tag}').")
        else:
            # 攻撃対象をランダムに選択
            indices_to_perturb = random.sample(target_indices, min(self.max_words, len(target_indices)))
            
            # トークンの表面形（単語）のリストを作成
            current_tokens = [t.surface for t in tokens]
            
            for idx in indices_to_perturb:
                original_word = current_tokens[idx]
                
                # --- マスク文の作成 ---
                # リストをコピーし、対象の単語だけを [MASK] に置き換える
                temp_tokens = current_tokens.copy()
                temp_tokens[idx] = self.mask_token
                masked_text = "".join(temp_tokens)
                
                # --- MLMによる予測 ---
                try:
                    # パイプラインで穴埋め候補を取得
                    candidates = self.fill_mask(masked_text)
                    
                    best_candidate = None
                    
                    # 候補の中から、元の単語と異なるものを選ぶ
                    for cand in candidates:
                        token_str = cand['token_str'].replace(' ', '') # 空白除去
                        
                        # 元の単語とも、マスク記号とも違う場合のみ採用
                        if token_str != original_word and token_str != self.mask_token:
                            best_candidate = token_str
                            break # 最もスコアが高い別の単語を採用
                    
                    if best_candidate:
                        print(f"[DEBUG-SENTENCE] -> Synonym Replace: '{original_word}' -> '{best_candidate}'")
                        # リストを更新
                        current_tokens[idx] = best_candidate
                    else:
                        print(f"[DEBUG-SENTENCE] -> No suitable synonym found for '{original_word}'")
                        
                except Exception as e:
                    print(f"[DEBUG-SENTENCE] -> MLM Error: {e}")
            
            # 文の再構築
            perturbed_text = "".join(current_tokens)
            
        # 結果を格納 (SR: Synonym Replacement)
        sample[f'{self.data_field}_perturbed_SR'] = perturbed_text
        
        # DEBUG: 最終結果を表示
        print(f"[DEBUG-SENTENCE] Final Perturbed Text: {perturbed_text}")
        print("----------------------------------")
        
        return sample

if __name__ == "__main__":
    print("="*50)
    print("   Synonym Replacement Test Runner")
    print("="*50 + "\n")

    # ダミーデータの準備
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': [''], 'context': ['']})
    
    # --- Test Case 1: 名詞の置換 (基本) ---
    print("\n--- Test Case 1: Noun Replacement (max_words=1) ---")
    test_sentence_1 = "日本の首相が、新しい研究開発の予算を決定した。"
    replacer_noun = SynonymReplace(data=DUMMY_DATA, data_field='question', max_words=1, pos_tag='名詞')
    
    dummy_sample_1 = DUMMY_DATA[0].copy()
    dummy_sample_1['question'] = test_sentence_1
    replacer_noun.apply_on_sample(dummy_sample_1)


    # --- Test Case 2: 動詞の置換 ---
    print("\n--- Test Case 2: Verb Replacement (max_words=2) ---")
    # "走る" -> "動く", "食べる" -> "飲む" など
    test_sentence_2 = "彼は公園で速く走り、ご飯を食べる。"
    # max_words=2 にして、文中の2つの動詞を狙う
    replacer_verb = SynonymReplace(data=DUMMY_DATA, data_field='question', max_words=2, pos_tag='動詞')
    
    dummy_sample_2 = DUMMY_DATA[0].copy()
    dummy_sample_2['question'] = test_sentence_2
    replacer_verb.apply_on_sample(dummy_sample_2)


    # --- Test Case 3: 形容詞の置換 ---
    print("\n--- Test Case 3: Adjective Replacement ---")
    # "美しい" -> "綺麗", "高い" -> "大きい" など
    test_sentence_3 = "この美しい花はとても高い。"
    replacer_adj = SynonymReplace(data=DUMMY_DATA, data_field='question', max_words=2, pos_tag='形容詞')
    
    dummy_sample_3 = DUMMY_DATA[0].copy()
    dummy_sample_3['question'] = test_sentence_3
    replacer_adj.apply_on_sample(dummy_sample_3)