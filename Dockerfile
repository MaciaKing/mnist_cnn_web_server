FROM python:3.11-slim

WORKDIR /app

# Copiar el contenido del directorio actual al contenedor
COPY . /app

# Instalar las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
