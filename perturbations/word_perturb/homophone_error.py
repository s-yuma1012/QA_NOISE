import random
import os
import requests
import jaconv
import re
from typing import Dict, List
from fugashi import Tagger 
from datasets import Dataset
from tqdm import tqdm

# MeCab/FugashiのTaggerをグローバルで初期化
tagger = Tagger()

class HomophoneError:
    """
    日本語版の漢字変換ミス攻撃クラス。
    SKK辞書を利用して、同音異義語（または同音の誤字）に置換します。
    例: "会議" -> "懐疑"
    """
    def __init__(self, data: Dataset, data_field: str='question', max_words: int=1, pos_tag: str=None):
        self.max_words: int = max_words
        self.pos_tag = pos_tag
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'homophone_error_jp'
        
        # SKK辞書の準備
        self.skk_dict = self._load_skk_dict()

    def _load_skk_dict(self) -> Dict[str, List[str]]:
        """
        GitHubからSKK辞書(L)の生テキストをダウンロードしてパースし、{'よみ': ['候補', ...]} の形式で返す
        """
        dict_path = "skk_dict.txt"
        # GitHubのRawファイルURL
        url = "https://raw.githubusercontent.com/skk-dev/dict/master/SKK-JISYO.L"
        
        # 1. 辞書がなければダウンロード
        if not os.path.exists(dict_path):
            print("[INFO] Downloading SKK Dictionary from GitHub...")
            try:
                response = requests.get(url)
                # SKK辞書はEUC-JP
                content = response.content.decode('euc-jp', errors='ignore')
                
                with open(dict_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                print(f"[ERROR] Failed to download dictionary: {e}")
                return {}

        # 2. 辞書の読み込みとパース
        skk_map = {}
        print("[INFO] Loading SKK Dictionary...")
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith(';;'): continue # コメント行スキップ
                    parts = line.strip().split(' ')
                    if len(parts) < 2: continue
                    
                    yomi = parts[0] # ひらがな
                    candidates_raw = parts[1] # /候補1/候補2/.../
                    
                    # /を除去してリスト化
                    candidates = [c for c in candidates_raw.split('/') if c]
                    
                    # 注釈（例: 会議;集まり）を除去して単語のみにする
                    clean_candidates = []
                    for c in candidates:
                        word = c.split(';')[0]
                        if word: 
                            clean_candidates.append(word)
                    
                    skk_map[yomi] = clean_candidates
        except Exception as e:
            print(f"[ERROR] Failed to read dictionary file: {e}")
            return {}
                
        print(f"[INFO] Dictionary loaded. {len(skk_map)} entries.")
        return skk_map

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]
        
        # DEBUG: 処理開始と元のテキストを表示
        print(f"\n[DEBUG-SENTENCE] Original Text: {raw_text}")

        tokens = list(tagger(raw_text)) 
        target_indices = [] 
        
        # 1. 攻撃対象の選定
        for i, token in enumerate(tokens):
            word_text = token.surface
            
            try:
                feature_str = str(token.feature)
                
                # --- 品詞(POS)の取得 ---
                if "pos1='" in feature_str:
                    start = feature_str.find("pos1='") + len("pos1='")
                    end = feature_str.find("'", start)
                    pos = feature_str[start:end]
                else:
                    pos = feature_str.split(',')[0]
                
                # --- 読み仮名(カタカナ)の取得 (正規表現修正版) ---
                reading_katakana = None
                
                # 正規表現で 'kana' (書字形) または 'pron' (発音形) の値を抽出
                # 例: kana='カイギ' -> カイギ を抽出
                match = re.search(r"kana='([^']*)'", feature_str)
                if match:
                    reading_katakana = match.group(1)
                else:
                    # kanaがなければpronを探す
                    match = re.search(r"pron='([^']*)'", feature_str)
                    if match:
                        reading_katakana = match.group(1)
                
                if not reading_katakana:
                    # 読みが取れなければスキップ
                    continue

            except:
                continue 

            # 助詞などは除外
            if pos in ['助詞', '助動詞', '記号', 'BOS/EOS', '空白']:
                continue
            
            # 品詞タグ設定と一致するかチェック
            if self.pos_tag is None or pos == self.pos_tag:
                # jaconvでカタカナ -> ひらがな変換 (SKK辞書検索用)
                yomi_hiragana = jaconv.kata2hira(reading_katakana)
                
                # 辞書にその読みが存在するかチェック
                if yomi_hiragana in self.skk_dict:
                    candidates = self.skk_dict[yomi_hiragana]
                    # 元の単語以外の候補がある場合のみ対象にする
                    if len(candidates) > 1:
                        target_indices.append((i, yomi_hiragana))

        # 2. 攻撃実行
        if not target_indices:
            perturbed_text = raw_text
            print(f"[DEBUG-SENTENCE] -> No targets found.")
        else:
            word_list = [t.surface for t in tokens]
            
            # ランダムに選択
            targets = random.sample(target_indices, min(self.max_words, len(target_indices)))
            
            # インデックス降順にソート (今回は置換なので順序は関係ないが、習慣として)
            targets.sort(key=lambda x: x[0], reverse=True) 
            
            for idx, yomi in targets:
                original_word = word_list[idx]
                candidates = self.skk_dict[yomi]
                
                # 元の単語を除外した候補リスト
                valid_candidates = [c for c in candidates if c != original_word]
                
                if valid_candidates:
                    # ランダムに誤変換候補を選択
                    new_word = random.choice(valid_candidates)
                    
                    print(f"[DEBUG-SENTENCE] -> Homophone Error: '{original_word}'({yomi}) -> '{new_word}'")
                    word_list[idx] = new_word
            
            perturbed_text = "".join(word_list)
            
        sample[f'{self.data_field}_perturbed_HOM'] = perturbed_text
        
        print(f"[DEBUG-SENTENCE] Final: {perturbed_text}")
        print("----------------------------------")
        
        return sample

if __name__ == "__main__":
    # ダミーデータの準備
    DUMMY_DATA = Dataset.from_dict({'id': ['0'], 'question': ['ダミー'], 'context': ['ダミー']})
    
    # テストしたい文のリスト
    test_sentences = [
        "重要な会議で質問をする。",          # 基本パターン (会議->懐疑, 質問->室紋など)
        "彼は自信を持って解答した。",        # 自信->自身, 解答->回答
        "機械を移動させる。",               # 機械->機会, 移動->異動
        "速く走る。",                       # 動詞・形容詞 (速く->早く)
        "意思決定のプロセス。",             # 意思->医師/遺志
        "コンピューターを操作する。",       # カタカナ語 (辞書にあれば変換されるかも)
        "東京大学。",                       # 固有名詞 (候補がない場合スキップされるか確認)
    ]
    
    # 攻撃インスタンスを作成 (max_wordsを多めにして、可能な限り変換させる設定)
    attacker = HomophoneError(data=DUMMY_DATA, data_field='question', max_words=3, pos_tag=None)
    
    print("\n" + "="*50)
    print("   Homophone Error Attack Test Runner")
    print("="*50 + "\n")

    for i, sentence in enumerate(test_sentences):
        print(f"--- Test Case {i+1} ---")
        
        # サンプルのコピーを作成して上書き
        dummy_sample = DUMMY_DATA[0].copy()
        dummy_sample['question'] = sentence
        
        # 攻撃実行
        result = attacker.apply_on_sample(dummy_sample)
        
        # 結果表示 (DEBUGログが出るので、ここではシンプルに比較)
        original = sentence
        perturbed = result.get('question_perturbed_HOM', 'Error')
        
        if original != perturbed:
            print(f"✅ SUCCESS: '{original}' -> '{perturbed}'")
        else:
            print(f"⏺️ NO CHANGE (No candidates or skipped)")
        
        print("\n")