FROM python:3.9
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD [ "python3", 'app.py', "--host=0.0.0.0"] 