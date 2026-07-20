FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install uv \
    && uv sync --locked

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]