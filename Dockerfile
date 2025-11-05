FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치 (의존성만)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    a2a-sdk>=0.3.5 \
    google-adk>=1.14.1 \
    google-genai>=1.36.0 \
    pydantic>=2.11.9 \
    python-dotenv>=1.1.1 \
    uvicorn>=0.35.0

# 환경 변수 설정
ENV PYTHONPATH=/app/src

# 포트 노출 (문서화용)
EXPOSE 9009 9018 9019

# 기본 쉘로 시작 (수동 실행)
CMD ["/bin/bash"]