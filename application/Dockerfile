FROM python:3.10-slim-bullseye

# Tiktoken requires Rust toolchain, so build it in a separate stage
RUN apt-get update && apt-get install -y gcc curl

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && apt-get install --reinstall libc6-dev -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --upgrade pip 

RUN apt-get update && apt-get install -y gnupg2

# Add Microsoft repository
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - 
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Update system and install packages
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools unixodbc-dev

COPY . /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY temp/tool.py /usr/local/lib/python3.10/site-packages/langchain/tools/sql_database/tool.py

WORKDIR /app
ENV FLASK_APP=app.py
ENV FLASK_DEBUG=true

EXPOSE 5010

CMD ["gunicorn", "-w", "2", "--timeout", "120", "--bind", "0.0.0.0:5010", "wsgi:app"]
