@echo off
set MY_PATH=.\ffmpeg\bin
set PATH=%PATH%;%MY_PATH%

runtime\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
pause