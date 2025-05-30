FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
# RUN sed -i 's/archive.ubuntu.com/hk.archive.ubuntu.com/g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y git git-lfs ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements/requirements.txt

ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860

CMD ["python", "app.py", "--version", "ppt-v2", "--checkpoint_dir", "checkpoints/ppt-v2", "--local_files_only" ]
