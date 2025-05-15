from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # maybe try yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large) cody said small is bad

model.train(
    data='/home/wrhamilt/project/School/CHEM_277B/final_project/tiles_output/tiles_output/yolo_dataset/data.yaml', 
    epochs=50,        
    imgsz=512,        
    batch=16,         
    name='yolov8s_canine_cancer_big',  
    pretrained=True    
)
