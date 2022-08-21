# Base image
FROM python:3.9
# Creation of a working directory app
WORKDIR /app
# Copy all the files of this project inside the container
COPY . .
# Installation of code dependencies
RUN pip install -r requirements.txt

# CAREFUL - use the special IP 0.0.0.0 inside a container
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5000"]
