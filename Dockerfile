FROM python:3.10

# Create a app directory
RUN mkdir /app
WORKDIR /app

# Copy over the app
COPY ./app /app

# Install dependencies
RUN pip install -r requirements.txt

# Set the default command
CMD ["streamlit", "run", "egu.py"]