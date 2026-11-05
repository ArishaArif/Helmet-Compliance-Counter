# from ultralytics import YOLO

# def main():
#     model = YOLO("yolov8n.pt")

#     model.train(
#         data=r"D:\HelmetComplianceCounter\dataset_split\data.yaml",
#         epochs=50,
#         imgsz=416,
#         batch=4,
#         device="cpu",
#         workers=0,
#         project=r"D:\HelmetComplianceCounter\runs",
#         name="helmet_yolov8n",
#         pretrained=True,
#         patience=10,
#         plots=True,
#         save=True
#     )

#     print("Training completed!")

# if __name__ == "__main__":
#     main()
from ultralytics import YOLO

def main():
    model = YOLO(r"D:\HelmetComplianceCounter\runs\helmet_yolov8n\weights\last.pt")

    model.train(resume=True)
    print("Training completed!")
if __name__ == "__main__":
    main()