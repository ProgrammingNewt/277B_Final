from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

model = YOLO('runs/detect/yolov8s_canine_cancer_big/weights/best.pt')


metrics = model.val(save=True, plots=True)


results_dir = metrics.save_dir
print(f"Plots and metrics saved in: {results_dir}")
