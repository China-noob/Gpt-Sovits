from GPT_SoVITS.inference_webui import inference
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train Gpt-Sovits -- XHT')
    parser.add_argument('--text', type=str,default="你好")
    parser.add_argument('--text_lang', type=str, default="zh")
    parser.add_argument('--ref_audio_path', type=str)
    parser.add_argument('--prompt_text', type=str)
    parser.add_argument('--rompt_lang', type=str,  default="zh")
    parser.add_argument('--top_k', type=int , default=5)
    parser.add_argument('--top_p', type=int , default=1)
    parser.add_argument('--temperature', type=int , default=1)
    parser.add_argument('--text_split_method', type=str,default="cut1")
    parser.add_argument('--batch_size', type=int,default=1)
    parser.add_argument('--speed_factor', type=float,default=1.0)
    parser.add_argument('--ref_text_free', type=bool,default=False)
    parser.add_argument('--split_bucket', type=bool,default=True)
    parser.add_argument('--fragment_interval', type=float,default=0.3)
    
    
    args = parser.parse_args()  
    
    sr,audio=inference(args.text,args.text_lang,args.ref_audio_path,args.prompt_text,args.rompt_lang,args.top_k,args.top_p,args.temperature,args.text_split_method,args.batch_size,args.speed_factor,args.ref_text_free,args.split_bucket,args.fragment_interval)
    print(sr)
    print(audio.shape)