FROM python:3.10-slim

# Set working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt .

# Upgrade pip and install dependencies.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code into the container.
COPY . .

# Expose Streamlit's default port.
EXPOSE 8501

# Run the Streamlit app. Disable CORS for simplicity.
CMD ["streamlit", "run", "streamlit_app.py", "--server.enableCORS", "false", "--server.port", "8501"]
