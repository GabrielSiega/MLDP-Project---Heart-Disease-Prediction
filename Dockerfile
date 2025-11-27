# Use Miniconda base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment.yml if you have one
COPY environment.yml .

# Install packages directly into base environment to avoid prefix issues
RUN conda install -y python=3.10 pandas numpy matplotlib seaborn scikit-learn plotly notebook \
    && pip install yellowbrick

# Copy your project notebooks and files
COPY . .

# Expose port for Jupyter
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
