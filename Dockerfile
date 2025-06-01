FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY app.py /app/
COPY model.py /app/
COPY best_model.pth /app/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]

#docker build -t my_classifier_api .
