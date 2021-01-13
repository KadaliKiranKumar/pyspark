FROM jupyter/pyspark-notebook
# FROM frolvlad/alpine-python-machinelearning

# COPY requirements.txt /
# RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app
CMD ["python", "project.py"]
