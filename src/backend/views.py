import numpy as np
import onnxruntime
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from PIL import Image

from backend.settings import BASE_DIR

imageClassList = {"0": "Man", "1": "Woman", "2": "Unknown"}  # Сюда указать классы


def scoreImagePage(request):
    return render(request, "scorepage.html")


def predictImage(request):
    fileObj = request.FILES["filePath"]
    fs = FileSystemStorage()
    filePathName = fs.save("images/" + fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    modelName = request.POST.get("modelName")
    scorePrediction = predictImageData(modelName, "./src" + filePathName)
    context = {"scorePrediction": scorePrediction}

    return render(request, "scorepage.html", context)


def predictImageData(modelName, filePath):
    print(filePath)
    print(BASE_DIR)
    img = Image.open(filePath).convert("RGB")
    img = np.asarray(img.resize((32, 32), Image.LANCZOS))
    sess = onnxruntime.InferenceSession(
        r"/Users/aspakalo/Projects/--educational/t5-2-/Neuroscience__/Homework1/src/resnet_avatars_gender/models/resnet_avatars_gender.onnx"
    )  # <-Здесь требуется указать свой путь к модели
    outputOFModel = np.argmax(
        sess.run(None, {"input": np.asarray([img]).astype(np.float32)})
    )
    score = imageClassList[str(outputOFModel)]

    return score
