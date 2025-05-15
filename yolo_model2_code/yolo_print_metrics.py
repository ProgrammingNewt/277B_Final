from ultralytics import YOLO

model = YOLO('runs/detect/yolov8s_canine_cancer_big/weights/best.pt')

metrics = model.val(save=True, plots=True)

precision = metrics.box.mp
recall = metrics.box.mr
map50 = metrics.box.map50
map95 = metrics.box.map
per_class_map = metrics.box.maps  # list of mAPs per class

print("\n Evaluation Metrics Summary:")
print(f"Precision      : {precision:.3f}")
print(f"Recall         : {recall:.3f}")
print(f"mAP@0.5        : {map50:.3f}")
print(f"mAP@0.5:0.95   : {map95:.3f}")

print("\n Per-Class mAP@0.5:")
for i, m in enumerate(per_class_map):
    class_name = model.names[i]
    print(f"{class_name:<20}: {m:.3f}")
