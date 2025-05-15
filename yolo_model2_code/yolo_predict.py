from ultralytics import YOLO

model = YOLO('runs/detect/yolov8s_canine_cancer_big/weights/best.pt')


metrics = model.val()

results = model.predict(source='./tiles_output/tiles_output/yolo_dataset/images/val/', save=True, imgsz=640)
