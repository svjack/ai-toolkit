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
