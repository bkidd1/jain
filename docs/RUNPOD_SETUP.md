# RunPod Setup for JAIN Experiments

Quick guide to run larger model extractions on RunPod.

## 1. Create Account

1. Go to [runpod.io](https://runpod.io)
2. Sign up with email or GitHub
3. Add credits ($10-25 is plenty)

## 2. Launch a Pod

**Recommended template:** `RunPod Pytorch 2.1`

**GPU options (pick one):**
| GPU | VRAM | Cost | Can Run |
|-----|------|------|---------|
| RTX 4090 | 24GB | ~$0.44/hr | 7B-13B models |
| A100 40GB | 40GB | ~$1.50/hr | 14B-32B models |
| A100 80GB | 80GB | ~$2.00/hr | 70B+ models |

For our needs (7B-14B), **RTX 4090 or A100 40GB** is ideal.

**Settings:**
- Container disk: 50GB (for model downloads)
- Volume disk: 20GB (for results)

## 3. Connect to Pod

Once running, click "Connect" → "Start Web Terminal" or use SSH:

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

## 4. Setup Environment

```bash
# Clone the repo
git clone https://github.com/bkidd1/jain.git
cd jain

# Create venv and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install additional deps for GPU
pip install accelerate bitsandbytes
```

## 5. Run Extractions

```bash
cd jain
source .venv/bin/activate

# 7B model (fits on 24GB GPU)
python experiments/02_divergence_detection/scripts/extract_with_cot.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --device cuda

# 14B model (needs 40GB+ GPU or 4-bit quantization)
python experiments/02_divergence_detection/scripts/extract_with_cot.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --device cuda

# With 4-bit quantization (for larger models on smaller GPUs)
python experiments/02_divergence_detection/scripts/extract_with_cot.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --device cuda \
  --load-in-4bit
```

## 6. Download Results

```bash
# On the pod - zip results
cd jain/experiments/02_divergence_detection/data/extractions
zip extractions.zip *.jsonl

# From your local machine
scp -P <port> root@<pod-ip>:/root/jain/experiments/02_divergence_detection/data/extractions/extractions.zip .
```

Or use the RunPod web file browser to download.

## 7. Stop the Pod

**Important:** Stop or terminate the pod when done to avoid charges!

RunPod → My Pods → Stop (or Terminate)

---

## Cost Estimate

| Task | Time | GPU | Cost |
|------|------|-----|------|
| 7B extraction (85 pairs) | ~15 min | RTX 4090 | ~$0.15 |
| 14B extraction (85 pairs) | ~30 min | A100 40GB | ~$0.75 |
| Full experiment suite | ~2 hrs | A100 40GB | ~$3.00 |

---

## Models to Run

Priority order for larger models:

1. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` - Reasoning model, 7B
2. `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` - Reasoning model, 14B  
3. `meta-llama/Llama-3.1-8B-Instruct` - Mainstream model
4. `mistralai/Mistral-7B-Instruct-v0.3` - Popular architecture
5. `Qwen/Qwen2.5-14B-Instruct` - Larger Qwen

---

## Troubleshooting

**Out of memory:** Use `--load-in-4bit` flag or get a bigger GPU

**Slow download:** Models are cached after first download. Consider keeping the pod running if doing multiple experiments.

**SSH issues:** Use the web terminal instead, it always works.
