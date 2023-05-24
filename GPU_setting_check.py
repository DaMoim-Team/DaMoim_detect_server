import torch
import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')  # Python 3.X에 맞게 수정
import cv2

print("OpenCV version:", cv2.__version__)
print("CUDA support:", "YES" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "NO")

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of available GPUs:", torch.cuda.device_count())
