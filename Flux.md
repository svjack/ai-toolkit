```bash
sudo apt-get update && sudo apt-get install git-lfs cbm ffmpeg

git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
# install torch first
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
pip install datasets

from datasets import load_dataset
from uuid import uuid1
import os
ds = load_dataset("svjack/Genshin_Impact_Varesa_Images_Captioned")
ds["train"]
path = "Genshin_Impact_Varesa_Images_Captioned"
os.makedirs(path, exist_ok=True)
for ele in ds["train"]:
    uuid_val = str(uuid1())
    uuid_img = os.path.join(path ,"{}.png".format(uuid_val))
    uuid_txt = os.path.join(path ,"{}.txt".format(uuid_val))
    with open(uuid_txt, "w") as f:
        f.write(ele["joy-caption"])
    ele["image"].save(uuid_img)

edit os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" in run.py

cp config/examples/train_lora_flux_24gb.yaml config

### edit

huggingface-cli login

python run.py config/train_lora_flux_24gb.yaml
```

```python
import torch
from diffusers import FluxPipeline
from datasets import load_dataset
from tqdm import tqdm
import os

# 初始化模型
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("Flux_Anime_Landscape_Lora/my_first_flux_lora_v1_000001500.safetensors")
pipe.enable_model_cpu_offload()

# 从 Hugging Face 加载数据集，并启用 streaming 模式
dataset = load_dataset("Falah/landscape_prompts", split="train", streaming=True)

# 设置输出文件夹路径
output_folder = "anime_generated_images"
os.makedirs(output_folder, exist_ok=True)

# 使用 tqdm 打印进度条并逐条处理
for idx, item in enumerate(tqdm(dataset, desc="Generating images")):
    prompt = item["prompts"]  # 假设字段名为 "prompt"
    prompt = "anime style ," + prompt

    # 生成图片
    try:
        image = pipe(prompt,
                     num_inference_steps=50,
                     guidance_scale=3.5,
                    ).images[0]
    except Exception as e:
        print(f"生成图像失败 (index={idx}): {e}")
        continue

    # 定义文件名（例如：image_0000.png）
    filename = f"image_{idx:04d}"
    image_path = os.path.join(output_folder, f"{filename}.png")
    text_path = os.path.join(output_folder, f"{filename}.txt")

    # 保存图片
    image.save(image_path)

    # 保存 prompt 到 .txt 文件
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(prompt)
```
