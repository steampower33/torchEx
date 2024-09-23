# 데이터셋은 scikit-learn에 있는 iris데이터셋을 불러오고,
# matplotlib으로 그래프를 2개 생성하는데, 
# 첫번째는 sepal width를 x축으로, sepal length를 y축으로 해서 그래프를 생성한다.
# 두번째는 petal width를 x축으로, petal length를 y축으로 해서 그래프를 생성한다.

# 데이터셋을 pytorch를 사용하여 신경망으로 클래스를 분류하는 코드를 생성한다
# 단, train데이터와 test데이터는 8:2로 나눠주고,클래스 3개는 train과 test에 적절하게 설정한다

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("bus.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model