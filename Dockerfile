FROM continuumio/anaconda3:latest
WORKDIR /cnn
RUN cat /etc/os-release && apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y tshark
COPY ./app /cnn/app
RUN ls -la
WORKDIR /cnn/app
RUN conda config --add channels conda-forge && conda config --set channel_priority strict
RUN conda install --file requirements.txt -y --quiet
# RUN mkdir -p /opt/notebooks && jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root
# EXPOSE 8888
