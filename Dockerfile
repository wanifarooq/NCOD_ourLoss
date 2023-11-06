FROM  nvcr.io/nvidia/pytorch:22.10-py3
# maybe we also have a requirements.txt file
COPY pathToYourRepository/requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

WORKDIR implementation
#depends on the wether you want to run in interactive mode or you want it to automate uncomment the following

#ENTRYPOINT ["python"]
#CMD ["train.py","-c", "configuration.json", "--lr_u","1","--percent","0.5"]

