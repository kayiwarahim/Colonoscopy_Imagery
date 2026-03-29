# Use Python 3.11 base image (TensorFlow compatible)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Expose port for Streamlit
EXPOSE 8080

# Run Streamlit app (choose streamlit.py as entry point)
CMD ["streamlit", "run", "streamlit.py", "--server.port=8080", "--server.address=0.0.0.0"]
