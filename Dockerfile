FROM python:3.10-slim

# 禁用代理变量
ENV HTTP_PROXY=""
ENV http_proxy=""
ENV HTTPS_PROXY=""
ENV https_proxy=""
ENV NO_PROXY="*"

WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# 配置 gdal include 路径
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# 复制全部项目文件
COPY . /app

# pip 会先在 wheelhouse 查找，若不存在再在线下载
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

EXPOSE 7860

CMD ["python", "Gradio_V11.py"]
