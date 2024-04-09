import os
import shutil
def save_sovits_latest_model(name,models_folder,outputdir):
    model_files = [f for f in os.listdir(models_folder) if f.startswith(name + "_")]
    print(model_files)
    latest_epoch = max([int(f.split("_")[-1].split(".")[0][1:]) for f in model_files])
    print(latest_epoch)
    for i in model_files:
        if i.endswith(f"{latest_epoch}.pth"):
            print(f"{i} found")
            latest_model = i
    #shutil.copy(os.path.join(models_folder, latest_model), os.path.join(outputdir, f"{name}_gpt_latest_model.pth"))
save_sovits_latest_model("test","SoVITS_weights",outputdir="SoVITS_weights")
