FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install paddlepaddle==3.2.0
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p uploads

EXPOSE 5000

CMD ["python", "app.py"]
