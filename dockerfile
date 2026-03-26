FROM public.ecr.aws/bitnami/spark:latest

USER root

RUN id -u bitnami >/dev/null 2>&1 || useradd -u 1001 -g 0 -M -d /opt/bitnami/spark bitnami

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 3. CORREÇÃO DE PERMISSÃO: Garante que o usuário bitnami consiga ler/escrever em /app
RUN chown -R 1001:1001 /app

# 4. VOLTAR PARA USUÁRIO COMUM (Segurança e compatibilidade)
USER 1001

# 5. GARANTIR QUE O PYTHON SEJA ENCONTRADO
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV PYTHONPATH="${PYTHONPATH}:/app"