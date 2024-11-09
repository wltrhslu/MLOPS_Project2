# Base image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the necessary files
COPY main.py main.py
COPY modules modules
COPY train.sh train.sh

# Set the entrypoint
ENTRYPOINT ["./train.sh"]