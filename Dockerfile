FROM graphcore/pytorch-jupyter:3.2.0-ubuntu-20.04-20230331

WORKDIR /app

COPY main.py requirements.txt ./

RUN pip3 install -U pip && pip3 install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]