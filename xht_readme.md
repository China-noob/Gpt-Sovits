--环境搭建

1：创建conda环境
## conda create -n gptsovits python=3.11

2: 激活环境
## conda activate gptsovits

3：安装依赖
## pip install -r requirements.txt

4：下载预训练模型和其他必须文件
ffmpeg.exe ffplay.exe ffprobe.exe
ffmpeg
├────bin/
│    ├────ffmpeg.exe
│    ├────ffplay.exe
│    └────ffprobe.exe
├────doc/
├────LICENSE
├────presets/
└────README.txt

pytorch_model.bin  pytorch_model.bin s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt s2D488k.pth s2G488k.pth
GPT_SoVITS\pretrained_models
├────.gitignore
├────chinese-hubert-base/
│    ├────config.json
│    ├────preprocessor_config.json
│    └────pytorch_model.bin
├────chinese-roberta-wwm-ext-large/
│    ├────config.json
│    ├────pytorch_model.bin
│    └────tokenizer.json
├────s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
├────s2D488k.pth
└────s2G488k.pth


mp_rank_00_model_states.pt
GPT_SoVITS\resemble_enhance\model_repo
└────enhancer_stage2/
│    ├────ds/
│    │    └────G/
│    │    │    ├────default/
│    │    │    │    └────mp_rank_00_model_states.pt
│    │    │    └────latest
│    └────hparams.yaml

smoke.ckpt  smoke-e15.ckpt
GPT_weights
├────smoke-e15.ckpt
└────smoke.ckpt

mp_rank_00_model_states.pt
resemble_enhance\model_repo\enhancer_stage2
├────ds/
│    └────G/
│    │    ├────default/
│    │    │    └────mp_rank_00_model_states.pt
│    │    └────latest
└────hparams.yaml

smoke.pth smoke_e8_s104.pth
SoVITS_weights
├────smoke.pth
└────smoke_e8_s104.pth
---------------------------------------------------------------------

--训练

## python .\train.py -n test -i D:\data\test -o D:\data\test

-n 实验名（用户id） -i 输入文件夹 -o 输出文件夹(与输入文件夹一样)
给我输入参数文件夹比如   D:\data\test  里面是wav格式音频文件

输出模型固定名字gpt.pth sovits.pth 文件夹目录为输入文件夹


---------------------------------------------------------------------
合成：
python inference.py -text -text_lang -ref_audio_path -prompt_text -rompt_lang -top_k -top_p -temperature -text_split_method -batch_size -speed_factor -ref_text_free -split_bucket -fragment_interval

-text 文本 （中日英）
-text_lang 文本语言（"中文",#全部按中文识别
                    "英文",#全部按英文识别
                    "日文",#全部按日文识别
                    "中英混合",#按中英混合识别
                    "日英混合",#按日英混合识别
                    "多语种混合",#多语种启动切分识别语种
                    ）
-ref_audio_path 参考音频路径
-prompt_text 参考音频文本
-rompt_lang 参考音频文本语言
-top_k 采样k 默认5
-top_p 采样p 默认1
-temperature 采样温度 默认1
-text_split_method 文本分割方法（"不切"
                                "凑四句一切"
                                "凑50字一切"
                                "按中文句号。切"
                                "按英文句号.切"
                                "按标点符号切"
                                )
-batch_size 批处理大小 默认1
-speed_factor 音频速度 默认1.0
-ref_text_free 是否开启无参考文本模式 默认false
-split_bucket 分割桶 默认 True
-fragment_interval 片段间隔 默认 0.3

例子python .\inference.py --ref_audio_path logs\test\5-wav32k\11.wav_0000040320_0000156160.wav_0000000000_0000116160.wav --prompt_text 我们今天讲第四章概要设计