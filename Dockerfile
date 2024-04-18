# Use the official Python image as the base image
FROM python:3.8

# Set the working directory in the container
# WORKDIR /app
WORKDIR . 

# Copy the application files into the working directory
# COPY . /app
COPY . .

# Install the application dependencies
RUN pip install -r requirements.txt

# Define the entry point for the container
# CMD ["flask", "run", "--host=0.0.0.0"]
CMD ["flask", "--app", "index run"]

