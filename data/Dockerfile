# ---------- build stage ---------- #
FROM python:3.11-slim AS builder
WORKDIR /app

# 安装 Python 依赖（包括 pyspark）
COPY requirements.txt .
RUN pip install --user -r requirements.txt && \
    python -m pip install --user pyspark==3.5.1

# 把 local 安装目录加入 PATH
ENV PATH=/root/.local/bin:$PATH

# ---------- runtime stage ---------- #
FROM openjdk:21-jdk-slim
LABEL maintainer="you@example.com"

# 允许在构建时指定 Spark/Hadoop 版本
ARG SPARK_VERSION=3.5.1
ARG HADOOP_VERSION=3

# 安装下载工具、procps (ps 命令) 以及 Python3 运行环境
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      procps \
      python3 && \
    rm -rf /var/lib/apt/lists/*

# 下载并解压 Spark 二进制包
RUN curl -Lfs \
      https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    | tar -xz -C /opt && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark

# 设置 Spark 环境变量
ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${PATH}"

# 将 builder 阶段安装的 Python 包复制过来
COPY --from=builder /root/.local /root/.local



# 指定 Spark 在 Driver 和 Executor 上都用 python3
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# 拷贝应用代码和数据
WORKDIR /app
COPY spark ./spark
COPY movielens ./data


# 启动命令：提交训练脚本
ENTRYPOINT ["spark-submit", "--master", "local[*]", "spark/train_item_embeddings.py"]
