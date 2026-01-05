# 使用 python:3.11-slim 作为基础镜像
FROM python:3.11-slim-bookworm

# 设置时区为中国标准时间
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 复制 uv 二进制文件
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装依赖
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码并安装项目
COPY . .
RUN uv sync --locked --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置默认命令
CMD ["uv", "run", "trade.py"]