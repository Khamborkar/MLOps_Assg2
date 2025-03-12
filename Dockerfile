FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install dvc
COPY . .
# CMD ["python", "src/train.py"]
