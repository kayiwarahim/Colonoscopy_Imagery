# Use official TensorFlow image (includes Python 3.11 + TF 2.22)
FROM tensorflow/tensorflow:2.21.0

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip and build tools first
RUN pip install --upgrade pip setuptools wheel

# Install only lightweight dependencies (TensorFlow is already included)
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt
# RUN pip install --ignore-installed -r requirements.txt

# Copy the rest of your project
COPY . .

# Expose Streamlit port
EXPOSE 8080

# Run Streamlit app (choose streamlit.py as entry point)
CMD ["streamlit", "run", "streamlit.py", "--server.port=8080", "--server.address=0.0.0.0"]