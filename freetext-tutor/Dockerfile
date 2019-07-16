FROM python:3.7

RUN apt-get update -y && apt-get install -y \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get install mongodb

WORKDIR /app

ADD requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN python -m spacy download de
# RUN python -m spacy download en

ADD . /app

# Expose the web port
EXPOSE 5000

# CMD ["gunicorn",  "-b", ":5000", "app:app"]
