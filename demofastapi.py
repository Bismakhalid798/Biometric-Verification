from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.encoders import jsonable_encoder
import os
import base64
import uuid
import numpy as np
import cv2
from enhanced import processImage
import uvicorn

app = FastAPI()

UPLOAD_PATH = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_enhanced_images(image: UploadFile = File(...)):
    image_array = np.frombuffer(image.file.read(), dtype=np.uint8)
    image_array = cv2.imdecode(image_array.astype(np.uint8), cv2.IMREAD_GRAYSCALE)
    enhanced_image = processImage(image_array)
    retval, buffer = cv2.imencode(".png", enhanced_image)
    image_data = buffer.tobytes()
    encoded_string = base64.b64encode(image_data).decode("utf-8")
    return encoded_string


@app.post("/multipleuploads")
async def upload(
    index: UploadFile = File(None),
    middle: UploadFile = File(None),
    ring: UploadFile = File(None),
    little: UploadFile = File(None),
) -> dict:
    enhanced_images = []
    if index:
        indexE = get_enhanced_images(index)
        enhanced_images.append({"image": indexE, "fingernumber": 2})
    if middle:
        middleE = get_enhanced_images(middle)
        enhanced_images.append({"image": middleE, "fingernumber": 3})
    if ring:
        ringE = get_enhanced_images(ring)
        enhanced_images.append({"image": ringE, "fingernumber": 4})
    if little:
        littleE = get_enhanced_images(little)
        enhanced_images.append({"image": littleE, "fingernumber": 5})
    return JSONResponse(content=jsonable_encoder({"enhanced_images": enhanced_images}))


if __name__ == "__main__":
    import asyncio

    uvicorn.run(
        "demofastapi:app", host="127.0.0.1", port=6003, log_level="info", reload=True
    )
