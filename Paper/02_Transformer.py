# fin_trans
# pip install transformers
# pip install sentencepiece

from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# 1. 번역 모델과 토크나이저 로드
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# 2. 소스 언어 설정
tokenizer.src_lang = "en"  # 소스 언어: 영어

# 3. 번역할 목표 언어 리스트
target_languages = ["ko", "ja", "fr"]  # 한국어, 일본어, 프랑스어

# 4. 번역할 텍스트
text = " The success of this plan hinges on three components: task, model, and data."

# 5. 각 목표 언어로 번역 수행
translations = {}
for target_lang in target_languages:
    # 목표 언어의 시작 토큰 ID 설정
    encoded_input = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
    
    # 번역 결과 저장
    translations[target_lang] = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# 6. 결과 출력
print(f"원문: {text}")
for lang, translation in translations.items():
    print(f"-" * 50)
    print(f"{lang} 번역: {translation}")




