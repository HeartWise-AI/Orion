FROM pytorch/pytorch:latest
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git nano -y
RUN pip install seaborn gdcm jupyter pandas scikit-learn scikit-image scikit-video watermark pydicom timesformer-pytorch
