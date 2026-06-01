from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == "__main__":

    freeze_support()

    model = YOLO(
        r"D:\Intern Project\YOLOv8_Detection-OperatorRooms\yolov8n.pt"
    )

    model.train(
        data=r"D:\Intern Project\YOLOv8_Detection-OperatorRooms\Dataset\data.yaml",
        epochs=110,
        imgsz=640,
        workers=0,
        batch=4,
        device=0,
        verbose=True
    )