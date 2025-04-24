```bash
sudo apt-get update && sudo apt-get install git-lfs cbm ffmpeg

git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
# install torch first
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
pip install datasets
pip install hf_xet
```

```python
from datasets import load_dataset
from uuid import uuid1
import os
ds = load_dataset("svjack/Cat_Boy_AMu_o_InstantCharacter_Images_Captioned_ShenheStyle")
ds["train"]
path = "Cat_Boy_AMu_o_InstantCharacter_Images_Captioned_ShenheStyle"
os.makedirs(path, exist_ok=True)
for ele in ds["train"]:
    uuid_val = str(uuid1())
    uuid_img = os.path.join(path ,"{}.png".format(uuid_val))
    uuid_txt = os.path.join(path ,"{}.txt".format(uuid_val))
    with open(uuid_txt, "w") as f:
        f.write(ele["prompt"])
    ele["image"].save(uuid_img)
```python

```bash
edit os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" in run.py

cp config/examples/train_lora_flex2_24gb.yaml config

python run.py config/train_lora_flex2_24gb.yaml
```
