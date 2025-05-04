from fastapi import FastAPI, UploadFile, File
import shutil, os
from pathlib import Path

app = FastAPI()

@app.post("/cartoonize")
async def cartoonize(file: UploadFile = File(...)):
    input_path = "inputs/input.jpg"
    output_path = "outputs/output.jpg"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    os.system(f"python3 test.py --checkpoint weights/face_paint_512_v2.pt --input {input_path} --output {output_path}")
    return {"status": "done", "output": output_path}
