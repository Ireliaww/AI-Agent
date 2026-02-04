# Multi-stage Dockerfile for AI-Agent
# Optimized for Google Cloud Run deployment

FROM python:3.11-slim as builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --user -r requirements.txt

# 生产阶段
FROM python:3.11-slim

WORKDIR /app

# 从builder复制已安装的包
COPY --from=builder /root/.local /root/.local

# 更新PATH
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY . .

# 创建非root用户（安全最佳实践）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口（Cloud Run会自动设置PORT环境变量）
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# 启动命令
CMD exec uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8080}
