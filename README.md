[🇺🇸 English](README_EN.md)

# 🇰🇷 LFM2-KoEn-Tuning

**LiquidAI LFM2-1.2B 기반 한국어-영어 양방향 번역 모델 파인튜닝**

[![Hugging Face](https://img.shields.io/badge/🤗%20Models-Hugging%20Face-yellow)](https://huggingface.co/gyung)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> ⚠️ **Note**: 해당 레포는 기존에 학습했던 코드들을 기반으로 안티그레비티(Antigravity)에서 Claude Opus 4.5를 써서 만든 코드들입니다. 아직 Colab과 Kaggle에서 실행을 하며 검증해보지 않았습니다. 해당 실험을 마치고 해당 링크를 포함해서 추가 업데이트를 할 예정입니다.

---

## 🏆 핵심 성과

> **1.2B 모델이 4B 모델을 압도!** Gemma-3 (4B)보다 **1.78 CHrF++** 높음  
> 단 **400 Step (0.78 Epoch)** 만에 SOTA 달성

---

## 📊 벤치마크 (Flores-200, 1012 Samples)

### 전체 모델 비교 (CHrF++ 기준 정렬)

| Rank | Model | CHrF++ | BLEU | Params | 비고 |
|:----:|-------|:------:|:----:|:------:|------|
| 1 | Google Translate | 39.27 | 18.18 | - | 상용 서비스 (Target) |
| 2 | Yanolja-4B-GGUF | 38.61 | 16.03 | 4B | Open Source SOTA |
| 3 | NLLB-200 (3.3B) | 35.09 | 11.68 | 3.3B | 번역 전용 모델 |
| **4** | **🏆 LFM2-v8-rl-10k-adapter** | **34.61** | **13.21** | **1.2B** | **본 프로젝트 SOTA** |
| 5 | LFM2-v6.4-merged | 33.53 | 12.31 | 1.2B | SFT Base |
| 6 | Gemma-3-4B-it-GGUF | 32.83 | 11.36 | 4B | Google 최신 4B |
| 7 | LFM2-v6.1-curriculum | 32.48 | 11.89 | 1.2B | SFT Curriculum |
| 8 | NLLB-200-Distilled-600M | 31.97 | 10.32 | 600M | 경량 번역 모델 |
| 9 | LFM2-v4-100k | 31.53 | 11.13 | 1.2B | 초기 SFT |
| 10 | LFM2-1.2B (Base) | 27.23 | 6.43 | 1.2B | 베이스라인 |
| 11 | Qwen3-4B-GGUF | 25.62 | 7.46 | 4B | Base Model |
| 12 | Gemma-3-1B-it-GGUF | 24.07 | 6.94 | 1B | 1B 모델 |
| 13 | Qwen3-1.7B-GGUF | 21.19 | - | 1.7B | Base Model |
| 14 | Qwen3-0.6B-GGUF | 13.48 | 1.98 | 0.6B | Base Model |

### GGUF 양자화 성능 (v8 merged 기준)

| Quantization | CHrF++ | BLEU | Size | 비고 |
|--------------|:------:|:----:|:----:|------|
| fp32 (원본) | 34.32 | 13.10 | 4.68G | 반복 버그 있음 |
| **Q8_0** 🏆 | **34.39** | 12.93 | 1.25G | 품질+안정성 최고 |
| Q5_K_M | 34.08 | 12.78 | 843M | 균형 추천 |
| Q4_K_M | 33.97 | 12.56 | 731M | 경량화/모바일 |

> **결론**: 4/5/8비트 양자화 모두 fp32와 사실상 동일한 성능!

---

## 📈 학습 과정별 성능 향상

| Step | Epoch | CHrF++ | BLEU | 비고 |
|:----:|:-----:|:------:|:----:|------|
| 0 | 0.00 | 33.53 | 12.63 | v6.4 Base |
| 200 | 0.39 | 34.10 | 12.93 | +0.57 향상 |
| 300 | 0.59 | 34.19 | 13.24 | Historic High |
| **400** | **0.78** | **34.61** | **13.21** | **🏆 SOTA** |

---

## ✨ v8 모델 강점

- **존댓말 일관성**: "합니다", "했습니다" 어미가 1012개 전체 샘플에서 일관 적용
- **자연스러운 문장**: 복잡한 문장도 자연스럽게 처리
- **문맥 인식**: "While"을 문맥에 따라 "반면", "동안" 등으로 유연하게 번역
- **전문 용어**: "rachis"를 "우축"으로 정확하게 번역

### ⚠️ 알려진 한계

- **고유명사 환각**: "George W. Bush" → "조지 워싱턴" (베이스 모델 편향)
- **해결 방안**: SFT + DPO를 통한 환각 교정 예정 (v9)

---

## 📂 프로젝트 구조
```
├── colab/              # Colab 노트북
│   ├── GRPO_v8_adapter_github.ipynb      # RL GRPO (SOTA)
│   ├── GRPO_v8_unsloth_vllm_github.ipynb # RL Unsloth+vLLM
│   ├── SFT_colab_github.ipynb            # SFT Colab 스타일 ⭐
│   └── SFT_v6.1_curriculum_github.ipynb  # SFT Kaggle 스타일
├── kaggle/             # Kaggle 노트북
│   ├── SFT_v6.1_curriculum.ipynb     # SFT v6.1
│   └── SFT_v6_200k.ipynb             # SFT v6 200k
├── evaluation/
│   └── benchmark_flores200.ipynb     # 벤치마크
├── quantization/
│   └── convert_to_gguf_github.ipynb  # GGUF 변환 (GitHub용)
└── dataset/
    ├── samples/                      # 학습 데이터 샘플
    └── upload_to_hf_github.py        # HF 업로드 스크립트 (GitHub용)
```

---

## 🚀 빠른 시작

### SOTA 모델 사용 (v8 Adapter)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "gyung/lfm2-1.2b-koen-mt-v6.4-merged",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("gyung/lfm2-1.2b-koen-mt-v6.4-merged")

# Adapter 로드 및 병합
model = PeftModel.from_pretrained(base_model, "gyung/lfm2-1.2b-koen-mt-v8-rl-10k-adapter")
model = model.merge_and_unload()

# 번역
messages = [
    {"role": "system", "content": "Translate to Korean."},
    {"role": "user", "content": "Hello, world!"}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=256, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### GGUF 사용 (llama.cpp)

```python
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    "gyung/lfm2-1.2b-koen-mt-v8-rl-10k-merged-GGUF",
    "lfm2-1.2b-koen-mt-v8-rl-10k-merged-Q8_0.gguf"
)

llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1)

prompt = """<|im_start|>system
Translate to Korean.<|im_end|>
<|im_start|>user
Hello, world!<|im_end|>
<|im_start|>assistant
"""
output = llm(prompt, max_tokens=256, stop=["<|im_end|>"], temperature=0.3)
print(output['choices'][0]['text'])
```

---

## 📊 데이터셋

`dataset/samples/`에 학습 데이터 샘플 포함:

| 파일 | 용도 | 개수 |
|------|------|------|
| `sample_sft_100_bidirectional.jsonl` | SFT | 100 |
| `sample_grpo_100_bidirectional.jsonl` | GRPO | 100 |

### HuggingFace 업로드

```bash
# .env 파일에 HF=your_token 설정 후
cd dataset
python upload_to_hf_github.py --repo YOUR_ID/your-dataset-name
```

---

## ⚙️ 학습 설정

### GRPO (v8 SOTA)

| 항목 | 값 |
|------|-----|
| Base Model | gyung/lfm2-1.2b-koen-mt-v6.4-merged |
| Method | GRPO (Group Relative Policy Optimization) |
| Reward | COMET + CHrF++ |
| Dataset | 10,000 samples (양방향) |
| Steps | 400 |
| LoRA Rank/Alpha | 32 / 64 |

### SFT (v6.4 Base)

```python
SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="paged_adamw_8bit",
    fp16=True,  # T4 최적화
)
```

---

## 🔗 모델 링크

| 모델 | 설명 | 링크 |
|------|------|------|
| **v8 Adapter** 🏆 | SOTA (CHrF++ 34.61) | [HuggingFace](https://huggingface.co/gyung/lfm2-1.2b-koen-mt-v8-rl-10k-adapter) |
| v8 GGUF | 양자화 버전 | [HuggingFace](https://huggingface.co/gyung/lfm2-1.2b-koen-mt-v8-rl-10k-merged-GGUF) |
| v6.4 Merged | Base 모델 | [HuggingFace](https://huggingface.co/gyung/lfm2-1.2b-koen-mt-v6.4-merged) |
| v4 100k | 초기 SFT | [HuggingFace](https://huggingface.co/gyung/lfm2-1.2b-koen-mt-v4-100k) |
| LFM2-1.2B | 원본 베이스 | [LiquidAI](https://huggingface.co/LiquidAI/LFM2-1.2B) |

---

## 📝 Citation

```bibtex
@misc{lfm2-koen-v8-rl,
  author = {gyung},
  title = {LFM2-1.2B-KoEn-MT: GRPO-Enhanced Korean-English Translation},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/gyung/lfm2-1.2b-koen-mt-v8-rl-10k-adapter}
}
```

---

## 📜 라이선스

이 모델은 **Liquid AI LFM Open License v1.0**을 따릅니다.

- ✅ 학술 연구 및 개인적 사용: 무제한
- ✅ 상업적 이용: 연 매출 $10M 미만 무료
- ⚠️ 연 매출 $10M 초과: 별도 라이선스 필요

---

*Last Updated: 2026-01-03*
