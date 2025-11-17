from transformers import AutoTokenizer

print("--- 1. 東北大モデル (日本語単一言語) ---")
tokenizer_ja = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3") 
print(f"「Amazon」(半角): {tokenizer_ja.tokenize('Amazon')}")
print(f"「Ａｍａｚｏｎ」(全角): {tokenizer_ja.tokenize('Ａｍａｚｏｎ')}")
print(f"「アメリカ」(カタカナ): {tokenizer_ja.tokenize('アメリカ')}")
print(f"「あめりか」(ひらがな): {tokenizer_ja.tokenize('あめりか')}")

print("\n--- 2. XLM-RoBERTa (多言語) ---")
tokenizer_xlm = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
print(f"「Amazon」(半角): {tokenizer_xlm.tokenize('Amazon')}")
print(f"「Ａｍａｚｏｎ」(全角): {tokenizer_xlm.tokenize('Ａｍａｚｏｎ')}")
print(f"「アメリカ」(カタカナ): {tokenizer_xlm.tokenize('アメリカ')}")
print(f"「あめりか」(ひらがな): {tokenizer_xlm.tokenize('あめりか')}")