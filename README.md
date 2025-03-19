# Open LLaVA-Video-R1

The current open-source code related to multimodal Deepseek-R1/GRPO is predominantly based on Qwen2VL.  However, in the field of video understanding, [LLaVA-Video](https://llava-vl.github.io/blog/2024-09-30-llava-video/), which serves as one of the most important baselines, still does not have any related open-source projects available (as of 2025/03/18). Therefore, we try to fill this gap by releasing a codebase, **Open-LLaVA-Video-R1**. 

## News
- [2025/03/19] We release the codebase of Open LLaVA-Video-R1

## What we did
To our best knowledge, we are the first to adapt R1/GRPO to LLaVA-Video architecture. In detail, we train [LLaVA-Video](https://llava-vl.github.io/blog/2024-09-30-llava-video/) using GRPO with accuracy and format rewards on the [DVD-counting](https://huggingface.co/datasets/Video-R1/DVD-counting) dataset. Training the 7B model on dvd datasets can be completed in approximately 5.5 hours using 8 x A800 (80G) GPUs. The training curve is as follows:

<img src="images\train.png" alt="7B_curve" style="zoom:90%;" />

## Performance
The experiment settting is the same as Qwen-based [Video-R1](https://github.com/tulerfeng/Video-R1), validated on the DVD-counting task. As shown in the Table, 11.5% gain is observed after using grpo training on [LLaVA-Video-Qwen](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2).

<div align="center">

| Dataset           | LLaVA-Video-7B | LLaVA-Video-7B+GRPO |
| ----------------- | -------------- | ------------------- |
| DVD-counting-test | 20.5           | **32.0 (11.5↑)**    |
</div>

### Set up
```
git clone https://github.com/Hui-design/Open-LLaVA-Video-R1.git
cd Open-LLaVA-Video-R1
```
Our environment is basically the same as [Open-r1-video](https://github.com/Wang-Xiaodong1899/Open-R1-Video) and [r1-video](https://github.com/tulerfeng/Video-R1). If you have already installed them, you can directly use the previous environment. If you haven't installed them yet, you can try the following commands.
```
conda create -n LLaVA-Video-R1 python=3.10
conda activate LLaVA-Video-R1
pip3 install -e ".[dev]"
pip3 install flash_attn --no-build-isolation
```

### Dataset 
We use the same task as [r1-video](https://github.com/tulerfeng/Video-R1), using the [DVD-counting](https://huggingface.co/datasets/Video-R1/DVD-counting) dataset.

Our dataset organization is:
```
dvd_dataset
  - dvd
    - *.mp4
  - train_dvd.jsonl
  - test_dvd.jsonl
```


### GRPO on LLaVA-Video
First download [LLaVA-Video-Qwen](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2), and modify the model_name_or_path in the train_llava_video.sh
```
# to run GRPO on llava_video
bash train_llava_video.sh
```


## Evaluation
Evaluation on video counting task

```bash
python llava_video_inference.py
```



## Acknowledgements

We sincerely appreciate the contributions of the open-source community. The related projects are as follows:

+ [Video-R1](https://github.com/tulerfeng/Video-R1)
+ [Open R1 Video](https://github.com/Wang-Xiaodong1899/Open-R1-Video)

  

