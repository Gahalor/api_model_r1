# Dockerfile
FROM python:3.11-slim

# Java para H2O MOJO + toolchain para SciPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Si prefieres paquete "genérico": default-jre-headless
# RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless build-essential && rm -rf /var/lib/apt/lists/*

# JAVA_HOME para JRE 17 en Debian
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads

ENV PORT=5320
EXPOSE 5320

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5320", "model:app", "--timeout", "120"]
