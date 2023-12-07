FROM python:3.8

WORKDIR /server

COPY . /server

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install git+https://github.com/fra31/auto-attack

RUN pip install git+https://github.com/RobustBench/robustbench.git
