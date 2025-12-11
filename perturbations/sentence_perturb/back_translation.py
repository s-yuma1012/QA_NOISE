import torch
from typing import Dict
from datasets import Dataset
from transformers import pipeline

class BackTranslation:
    
    def __init__(self, data: Dataset, data_field: str='question', device: int=None):
        self.data: Dataset = data
        self.data_field = data_field.lower() 
        self.name: str = 'back_translation_jp'
        
        # GPU設定 (-1: CPU, 0以上: GPU)
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
            
        print(f"[INFO] Loading Translation Pipelines on device {self.device}...")
        
        # 軽量モデル (Helsinki-NLP) をロード
        # 1. 日->英 (~300MB)
        print("[INFO] Loading JA -> EN model...")
        self.translator_ja_en = pipeline("translation", model="Helsinki-NLP/opus-mt-jap-en", device=self.device)
        
        # 2. 英->日 (~300MB)
        print("[INFO] Loading EN -> JA model...")
        self.translator_en_ja = pipeline("translation", model="Helsinki-NLP/opus-mt-en-jap", device=self.device)

    def apply_on_sample(self, sample: Dict) -> Dict:
        raw_text = sample[self.data_field]
        
        print(f"Original Text: {raw_text}")
        
        try:
            # 1. 日本語 -> 英語
            translated_en = self.translator_ja_en(raw_text, max_length=512)[0]['translation_text']
            print(f"-> Intermediate (EN): {translated_en}")
            
            # 2. 英語 -> 日本語
            back_translated_ja = self.translator_en_ja(translated_en, max_length=512)[0]['translation_text']
            
            perturbed_text = back_translated_ja
            
        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            perturbed_text = raw_text # 失敗時は元文を返す

        # 結果を格納
        sample[f'{self.data_field}_perturbed_BT'] = perturbed_text

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
    
    print(f"\n=== Back Translation Test (Helsinki-NLP) ===")
    
    attacker = BackTranslation(data=DUMMY_DATA, data_field='question')
    
    for i, sent in enumerate(test_sentences):
        dummy_sample = DUMMY_DATA[0].copy()
        dummy_sample['question'] = sent
        
        attacker.apply_on_sample(dummy_sample)