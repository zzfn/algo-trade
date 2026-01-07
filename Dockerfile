# 使用 python:3.11-slim 作为基础镜像
FROM python:3.11-slim

# 设置时区为中国标准时间
ENV TZ=Asia/Shanghai
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
# libgomp1 is required for lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制 uv 二进制文件
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装依赖
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

# 复制项目代码并安装项目
COPY . .
RUN uv sync

EXPOSE 8000

# 设置默认命令
CMD ["uv", "run","main.py"]