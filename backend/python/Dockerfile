FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]