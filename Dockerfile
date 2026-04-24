FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.10-bookworm

WORKDIR /code

RUN pip install uv -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY pyproject.toml uv.lock /code/

RUN uv sync --frozen --no-dev --no-install-project

COPY ./app /code/app
COPY ./nn_models /code/nn_models

CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]