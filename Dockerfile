# 使用 python:3.11-slim 作为基础镜像
FROM python:3.11-slim

# 设置时区为中国标准时间
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 复制 uv 二进制文件
COPY --from=ghcr.io/astral-sh/uv:0.8.9 /uv /uvx /bin/

# 可选：配置清华大学镜像源（加速 apt-get，国内用户推荐）
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main" > /etc/apt/sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian-security/ bookworm-security main" >> /etc/apt/sources.list

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装依赖
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# 复制项目代码并安装项目
COPY . .
RUN uv sync --locked

# 设置默认命令
CMD ["uv", "run", "trade.py"]