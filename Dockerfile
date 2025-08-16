FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install gradio
EXPOSE 7860
CMD ["python", "src/app.py"]
