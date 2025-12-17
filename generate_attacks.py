import argparse
import os
import torch
import json  # ★追加: 標準ライブラリを使用
from datasets import load_dataset, Dataset
from tqdm import tqdm

# --- 1. Import Attacker Classes ---

# === 文字レベル (Character Level) ===
from perturbations.char_perturb.delete_char import DeleteChar
from perturbations.char_perturb.insert_char import InsertChar
from perturbations.char_perturb.replace_char import ReplaceChar
from perturbations.char_perturb.repeat_char import RepeatChar
from perturbations.char_perturb.hiragana_to_katakana_char import HiraganaToKatakana as H2K_Char
from perturbations.char_perturb.katakana_to_hiragana_char import KatakanaToHiragana as K2H_Char

# === 単語レベル (Word Level) ===
from perturbations.word_perturb.replace_synonym import SynonymReplace
from perturbations.word_perturb.homophone_error import HomophoneError
from perturbations.word_perturb.swap_word import SwapWord
from perturbations.word_perturb.delete_word import DeleteWord
from perturbations.word_perturb.repeat_word import RepeatWord
from perturbations.word_perturb.hiragana_to_katakana_word import HiraganaToKatakana as H2K_Word
from perturbations.word_perturb.katakana_to_hiragana_word import KatakanaToHiragana as K2H_Word

# === 文レベル (Sentence Level) ===
from perturbations.sentence_perturb.back_translation import BackTranslation

# --- 2. Attack Registry ---
ATTACK_REGISTRY = {
    # Char Level
    'delete_char': DeleteChar,
    'insert_char': InsertChar,
    'replace_char': ReplaceChar,
    'repeat_char': RepeatChar,
    'hira2kata_char': H2K_Char,
    'kata2hira_char': K2H_Char,
    
    # Word Level
    'synonym_replace': SynonymReplace,
    'homophone_error': HomophoneError,
    'swap_word': SwapWord,
    'delete_word': DeleteWord,
    'repeat_word': RepeatWord,
    'hira2kata_word': H2K_Word,
    'kata2hira_word': K2H_Word,
    
    # Sentence Level
    'back_translation': BackTranslation
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate noisy datasets using various perturbation methods.")
    
    parser.add_argument(
        '--attacks', 
        nargs='+', 
        default=['all'], 
        help=f"List of attacks to generate. Options: {list(ATTACK_REGISTRY.keys())} or 'all'"
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./perturbed_data', 
        help="Directory to save the perturbed datasets."
    )
    
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        default='shunk031/JGLUE', 
        help="Hugging Face dataset path."
    )
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='JSQuAD', 
        help="Dataset configuration name."
    )
    parser.add_argument(
        '--split', 
        type=str, 
        default='validation', 
        help="Dataset split to use."
    )
    parser.add_argument(
        '--target_column', 
        type=str, 
        default='question', 
        help="The column name to apply perturbations to."
    )
    
    parser.add_argument(
        '--max_samples', 
        type=int, 
        default=None, 
        help="For testing: limit the number of samples to process."
    )

    return parser.parse_args()

def get_attacker_instance(attack_name, attack_class, dataset, device, target_column):
    # 共通設定
    common_args = {
        'data': dataset,
        'data_field': target_column
    }

    # クラスごとの特有パラメータ分岐
    if attack_name == 'back_translation':
        return attack_class(**common_args, device=device)
        
    elif 'char' in attack_name:
        if attack_name in ['hira2kata_char', 'kata2hira_char']:
             return attack_class(**common_args, max_words=2)
        else:
            return attack_class(**common_args, max_perturbs=2, length_of_word_to_perturb=1)
            
    elif 'word' in attack_name or 'synonym' in attack_name or 'homophone' in attack_name:
        return attack_class(**common_args, max_words=1, pos_tag=None)
        
    else:
        return attack_class(**common_args)

def main():
    args = parse_args()
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"[INFO] Running on device: {'GPU' if device != -1 else 'CPU'}")

    # 1. データセットのロード
    print(f"[INFO] Loading dataset: {args.dataset_path} (config: {args.dataset_name}, split: {args.split})...")
    
    try:
        config_name = None if args.dataset_name == 'None' else args.dataset_name
        raw_dataset = load_dataset(args.dataset_path, name=config_name, split=args.split)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    if args.target_column not in raw_dataset.column_names:
        print(f"[ERROR] Target column '{args.target_column}' not found.")
        return

    if args.max_samples:
        print(f"[INFO] Slicing dataset to first {args.max_samples} samples.")
        raw_dataset = raw_dataset.select(range(args.max_samples))

    if 'all' in args.attacks:
        target_attacks = list(ATTACK_REGISTRY.keys())
    else:
        target_attacks = args.attacks

    print(f"[INFO] Target Attacks: {target_attacks}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 攻撃ループ
    for attack_name in tqdm(target_attacks, desc="Processing Attacks"):
        print(f"\n" + "="*50)
        print(f"   Executing Attack: {attack_name}")
        print(f"="*50)
        
        try:
            attack_cls = ATTACK_REGISTRY[attack_name]
            attacker = get_attacker_instance(attack_name, attack_cls, raw_dataset, device, args.target_column)
            
            # データセットへの適用
            perturbed_dataset = raw_dataset.map(attacker.apply_on_sample, batched=False)
            
            # --- 修正箇所: 標準jsonライブラリを使って確実に保存 ---
            save_filename = f"{attack_name}.json"
            save_path = os.path.join(args.output_dir, save_filename)
            
            print(f"[INFO] Saving {attack_name} dataset to {save_path}...")
            
            # DatasetオブジェクトをPythonのリスト(dictの配列)に変換
            data_list = [sample for sample in perturbed_dataset]
            
            # json.dumpで書き込み (確実に1つの配列になる)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=4)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {attack_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n[INFO] All requested attacks completed.")

if __name__ == "__main__":
    main()