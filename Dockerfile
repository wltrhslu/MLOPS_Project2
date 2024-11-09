# Base image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# Copy the necessary files
COPY main.py main.py
COPY modules modules
COPY train.sh train.sh

# Make the script executable
RUN chmod +x train.sh

# Set the entrypoint
ENTRYPOINT ["./train.sh"]