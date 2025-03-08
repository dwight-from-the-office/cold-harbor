import torch
import cv2
import gymnasium
import ultralytics
import stable_baselines3



def main():
    print("Project environment verification:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Torch CUDA Available: {torch.cuda.is_available()}")
    print("Setup Successful!")

if __name__ == "__main__":
    main()