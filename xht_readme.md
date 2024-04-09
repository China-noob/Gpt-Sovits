--环境搭建

1：创建conda环境
## conda create -n gptsovits python=3.11

2: 激活环境
## conda activate gptsovits

3：安装依赖
## pip install -r requirements.txt

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
-text_lang 文本语言（"all_zh",#全部按中文识别
                    "en",#全部按英文识别
                    ”all_ja",#全部按日文识别
                    "zh",#按中英混合识别
                    "ja",#按日英混合识别
                    "auto",#多语种启动切分识别语种
                    ）
-ref_audio_path 参考音频路径
-prompt_text 参考音频文本
-rompt_lang 参考音频文本语言
-top_k 采样k 默认5
-top_p 采样p 默认1
-temperature 采样温度 默认1
-text_split_method 文本分割方法（("不切"):"cut0",
                                ("凑四句一切"): "cut1",
                                ("凑50字一切"): "cut2",
                                ("按中文句号。切"): "cut3"
                                ("按英文句号.切"): "cut4",
                                ("按标点符号切"): "cut5",）
-batch_size 批处理大小 默认1
-speed_factor 音频速度 默认1.0
-ref_text_free 是否开启无参考文本模式 默认false
-split_bucket 分割桶 默认 True
-fragment_interval 片段间隔 默认 0.3