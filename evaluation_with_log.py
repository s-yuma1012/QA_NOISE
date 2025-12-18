import argparse
import json
import os
import re
import string
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
import emoji
import neologdn
from fugashi import Tagger
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# === 1. User's Normalization Logic ===
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_emoji(text):
        text = "".join(["" if emoji.is_emoji(c) else c for c in text])
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text)

    return white_space_fix(neologdn.normalize(remove_emoji(s)))

def remove_punc(tokens):
    exclude = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    exclude += string.punctuation
    exclude = [*exclude]
    return [tok for tok in tokens if tok not in exclude]

# === 2. Metric Functions ===
def f1_score(prediction, ground_truth, tagger):
    prediction_tokens = remove_punc(tagger.parse(normalize_answer(prediction)).split())
    ground_truth_tokens = remove_punc(tagger.parse(normalize_answer(ground_truth)).split())
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, tagger=None):
    if metric_fn.__name__ == 'f1_score':
        scores = [metric_fn(prediction, gt, tagger) for gt in ground_truths]
    else:
        scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores)

# === 3. Evaluation Logic ===
def evaluate_dataset(model, tokenizer, tagger, dataset, device, batch_size, target_col):
    """
    1つのデータセット(JSON)に対する評価を実行する
    詳細な予測結果(predictions_list)も返すように変更
    """
    model.eval()
    
    total_f1 = 0
    total_em = 0
    count = 0
    
    # ★追加: 全サンプルの予測結果を保存するリスト
    detailed_predictions = []
    
    # バッチ処理
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Eval ({target_col})", leave=False):
        batch = dataset[i:i+batch_size]
        
        # 質問文と文脈の抽出
        questions = [entry[target_col] for entry in batch]
        contexts = [entry['context'] for entry in batch]
        
        # トークナイズ
        inputs = tokenizer(
            questions, 
            contexts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # デコードとスコア計算
        for j, entry in enumerate(batch):
            start_logits = outputs.start_logits[j]
            end_logits = outputs.end_logits[j]
            
            answer_start = torch.argmax(start_logits)
            answer_end = torch.argmax(end_logits) + 1
            
            prediction = tokenizer.decode(inputs["input_ids"][j][answer_start:answer_end], skip_special_tokens=True)
            
            # 正解データの取得
            raw_answers = entry['answers']
            if isinstance(raw_answers, dict) and 'text' in raw_answers:
                gold_answers = raw_answers['text']
            elif isinstance(raw_answers, list):
                gold_answers = raw_answers
            else:
                gold_answers = [str(raw_answers)]

            # スコア計算
            f1 = metric_max_over_ground_truths(f1_score, prediction, gold_answers, tagger)
            em = metric_max_over_ground_truths(exact_match_score, prediction, gold_answers)

            total_f1 += f1
            total_em += em
            count += 1
            
            # ★追加: 詳細結果の保存
            detailed_predictions.append({
                "id": entry.get('id', ''),      # IDがあれば追跡しやすい
                "question": questions[j],       # 入力した質問文（ノイズ入りかもしれない）
                "prediction": prediction,       # モデルの予測
                "ground_truths": gold_answers,  # 正解リスト
                "em": em,                       # そのサンプルのEMスコア(0 or 1)
                "f1": f1                        # そのサンプルのF1スコア(0.0 - 1.0)
            })
            
    avg_f1 = (total_f1 / count) * 100
    avg_em = (total_em / count) * 100

    return avg_f1, avg_em, detailed_predictions # ★リストも返す

# === 4. Main Controller ===
def main():
    parser = argparse.ArgumentParser(description="Evaluate QA models on perturbed datasets.")
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help="Hugging Face model path."
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='./perturbed_data', 
        help="Directory containing the perturbed json files."
    )
    
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='evaluation_results.json', 
        help="File to save the evaluation summary."
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help="Inference batch size."
    )

    parser.add_argument(
        '--force_clean',
        action='store_true',
        help="If set, evaluate the original 'question' column instead of the perturbed one."
    )
    
    # ★追加: 予測結果を保存するフォルダ指定
    parser.add_argument(
        '--pred_output_dir',
        type=str,
        default='./prediction_logs',
        help="Directory to save detailed prediction logs."
    )

    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Tagger, Model, Tokenizerの初期化
    print("[INFO] Loading Tokenizer & Model...")
    tagger = Tagger("-Owakati")
    # use_fast=False を追加（XLM-R対応のため）
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    model.to(device)

    results = []
    
    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Directory not found: {args.data_dir}")
        return

    # ★追加: 予測ログ用ディレクトリの作成
    if not os.path.exists(args.pred_output_dir):
        os.makedirs(args.pred_output_dir)

    # ファイル探索
    files = [f for f in os.listdir(args.data_dir) if f.endswith('.json') and not f.startswith('state')]
    files.sort()
    
    print(f"[INFO] Found {len(files)} datasets: {files}")
    
    if args.force_clean:
        print("[INFO] Evaluation Mode: CLEAN BASELINE (Evaluating once)")

    for filename in files:
        file_path = os.path.join(args.data_dir, filename)
        
        try:
            # データ読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # カラム特定
            first_sample = data[0]
            target_question_key = 'question'
            
            if args.force_clean:
                target_question_key = 'question'
            else:
                for key in first_sample.keys():
                    if key.startswith('question_perturbed_'):
                        target_question_key = key
                        break
            
            print(f"\nProcessing {filename} | Column: {target_question_key}")
            
            # 評価実行 (★戻り値としてpredictionsを受け取る)
            f1, em, predictions = evaluate_dataset(
                model, tokenizer, tagger, data, device, args.batch_size, target_question_key
            )
            
            print(f" -> EM: {em:.2f} | F1: {f1:.2f}")
            
            results.append({
                "filename": filename,
                "attack_type": target_question_key,
                "em": em,
                "f1": f1,
                "num_samples": len(data)
            })
            
            # ★追加: 詳細な予測結果をJSONとして保存
            pred_filename = f"preds_{filename}" # 例: preds_delete_char.json
            if args.force_clean:
                pred_filename = "preds_clean_baseline.json"
                
            pred_save_path = os.path.join(args.pred_output_dir, pred_filename)
            
            with open(pred_save_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=4)
            # print(f"    (Saved detailed predictions to {pred_save_path})")

            # クリーンモードなら1回で終了
            if args.force_clean:
                break
                
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {filename}: {e}")
            import traceback
            traceback.print_exc()

    # 結果サマリの保存
    save_file = args.output_file
    if args.force_clean:
        base, ext = os.path.splitext(save_file)
        save_file = f"{base}_clean{ext}"

    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n[INFO] Evaluation complete. Results saved to {save_file}")
    print(f"[INFO] Detailed predictions saved in: {args.pred_output_dir}")

if __name__ == "__main__":
    main()