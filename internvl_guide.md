# 📌 HƯỚNG DẪN CHẠY INTERNVL (INFERENCE & TRAINING)

---

## ✅ 1. Cài đặt môi trường
```bash
git clone https://github.com/OpenGVLab/InternVL.git
cd InternVL
conda create -n internvl python=3.9 -y
conda activate internvl
pip install -r requirements.txt
```

---
## ✅ 2. Cài phụ thuộc mở rộng
```bash
pip install flash-attn==2.3.6 --no-build-isolation
pip install -U openmim
mim install mmcv-full==1.6.2
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 2386a91
pip install -v --no-build-isolation ./
```

---
## ✅ 3. Inference với InternVL 3.0
```bash
pip install -U huggingface_hub
cd pretrained/
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-1B --local-dir InternVL3-1B
```

---
## ✅ 4. Fine-tune InternVL 3.0 (Custom Dataset)
**Bước 1: metadata JSON**
```json
{
  "my_dataset": {
    "root": "data/my_dataset/images/",
    "annotation": "data/my_dataset/annotations.jsonl",
    "data_augment": false,
    "max_dynamic_patch": 12,
    "repeat_time": 1,
    "length": 1000
  }
}
```
**Bước 2: Fine-tune**
```bash
# Full (8 GPU)
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh
# LoRA (2 GPU)
GPUS=2 sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh
```

---
## ✅ 5. Fine-tune InternVL 1.0 (Stage 2)
**Bước 1: Dữ liệu COCO**
```bash
mkdir -p data/coco/annotations && cd data/coco
wget http://images.cocodataset.org/zips/{train2014,val2014,test2015}.zip && unzip '*.zip'
cd annotations
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json
wget https://github.com/OpenGVLab/InternVL/releases/download/data/coco_karpathy_test.json
```
**Bước 2: Thêm custom dataset**
```python
ds_collections = {
  'my_dataset_coco_format': {
    'root': './data/my_dataset/',
    'annotation': './data/my_dataset/annotations.json',
  }
}
```
**Bước 3: Chạy fine-tune**
| Loại | Lệnh | Yêu cầu GPU |
|------|------|-------------|
| Full | `sh shell/finetune/internvl_stage2_finetune_coco_364_bs1024_ep5.sh` | 32×A100 (80G) |
| Head | `sh shell/head_finetune/internvl_stage2_finetune_coco_224_bs1024_ep5_head_4gpu.sh` | 4 GPU ≥32G |
| LoRA | `sh shell/lora_finetune/internvl_stage2_finetune_coco_224_bs1024_ep5_lora16_4gpu.sh` | 4 GPU ≥40G |

---
## ✅ 6. Inference / Evaluation
### Captioning
```bash
sh evaluate.sh pretrained/InternVL-14B-224px caption-coco
```
### Retrieval
```bash
cd clip_benchmark
CUDA_VISIBLE_DEVICES=0 python3 clip_benchmark/cli.py eval \
  --model_type internvl --language en --task zeroshot_retrieval \
  --dataset flickr30k --dataset_root ./data/flickr30k \
  --model internvl_c_retrieval_hf \
  --pretrained ./work_dirs/internvl_stage2_finetune_flickr_364_bs1024_ep10/ \
  --output result_ft.json
```

---
## 📚 Tài nguyên chính thức
- GitHub: https://github.com/OpenGVLab/InternVL
- HF models: https://huggingface.co/OpenGVLab