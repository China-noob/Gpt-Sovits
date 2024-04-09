import os,sys
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS"%(now_dir))
import argparse
from TTS_infer_pack.TTS import TTS, TTS_Config
import os,sys,time
import torch
from tools.i18n.i18n import I18nAuto
import numpy as np
from scipy.io.wavfile import write


i18n = I18nAuto()
dict_language = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}

cut_method = {
    i18n("不切"):"cut0",
    i18n("凑四句一切"): "cut1",
    i18n("凑50字一切"): "cut2",
    i18n("按中文句号。切"): "cut3",
    i18n("按英文句号.切"): "cut4",
    i18n("按标点符号切"): "cut5",
}


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and not torch.backends.mps.is_available()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def loadmodel(name):
    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = device
    tts_config.is_half = is_half
   
    tts_config.t2s_weights_path=os.path.join("GPT_weights",name+".ckpt")
    tts_config.vits_weights_path=os.path.join("SoVITS_weights",name+".pth")
    tts_pipline = TTS(tts_config)
    return tts_pipline
    
def inference(text, text_lang, 
              ref_audio_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket,fragment_interval,
              tts_pipline
              ):
    start_time = time.time()  # 记录开始时间
    inputs={
        "text": text,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size":int(batch_size),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "return_fragment":False,
        "fragment_interval":fragment_interval,
    }

    yield next(tts_pipline.run(inputs))

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"推理 executed in {elapsed_time:.6f} seconds")  # 输出执行时间，保留6位小数  
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train Gpt-Sovits -- XHT')
    parser.add_argument('--text', type=str,default="你好")
    parser.add_argument('--text_lang', type=str, default="中文")
    parser.add_argument('--ref_audio_path', type=str)
    parser.add_argument('--prompt_text', type=str)
    parser.add_argument('--rompt_lang', type=str,  default="中文")
    parser.add_argument('--top_k', type=int , default=5)
    parser.add_argument('--top_p', type=int , default=1)
    parser.add_argument('--temperature', type=int , default=1)
    parser.add_argument('--text_split_method', type=str,default="凑四句一切")
    parser.add_argument('--batch_size', type=int,default=1)
    parser.add_argument('--speed_factor', type=float,default=1.0)
    parser.add_argument('--ref_text_free', type=bool,default=False)
    parser.add_argument('--split_bucket', type=bool,default=True)
    parser.add_argument('--fragment_interval', type=float,default=0.3)
    parser.add_argument('--name', type=str,default="smoke")
    
    
    args = parser.parse_args()  
    
    tts_pipline=loadmodel(args.name) 
    data=inference(args.text,args.text_lang,args.ref_audio_path,args.prompt_text,args.rompt_lang,args.top_k,args.top_p,args.temperature,args.text_split_method,args.batch_size,args.speed_factor,args.ref_text_free,args.split_bucket,args.fragment_interval,tts_pipline)
  
    audio=next(data)[1]
   

    # 保存为 WAV 文件
    write("inference_test.wav", 32000, audio) 