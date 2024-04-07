## Cli-demo

![image-20240403165548862](hw.assets/image-20240403165548862.png)



# 进阶作业

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

## lagent

![image-20240407001515566](hw.assets/image-20240407001515566.png)



![image-20240407001456713](hw.assets/image-20240407001456713.png)

## Hugging Face

```bash
export HF_ENDPOINT=https://hf-mirror.com

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```



![image-20240407094949943](hw.assets/image-20240407094949943.png)



## 浦语 灵笔

### 1. 图文创作

```
python /root/demo/InternLM-XComposer/examples/gradio_demo_composition.py --code_path /root/models/internlm-xcomposer2-7b --private --num_gpus 1 --port 6006
```



![image-20240407110533525](hw.assets/image-20240407110533525.png)

```
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 41783

nLHFNnx5lSaY1i1r
```

![image-20240407110741394](hw.assets/image-20240407110741394.png)

![image-20240407111017047](hw.assets/image-20240407111017047.png)

### 2. 视觉问答

```
cd /root/demo/InternLM-XComposer
python /root/demo/InternLM-XComposer/examples/gradio_demo_chat.py  --code_path /root/models/internlm-xcomposer2-vl-7b --private --num_gpus 1 --port 6006
```

![image-20240407120253262](hw.assets/image-20240407120253262.png)