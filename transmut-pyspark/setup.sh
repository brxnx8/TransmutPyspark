#!/bin/bash

echo "Preparando o ambiente Linux para o PySpark..."

sudo apt update && sudo apt upgrade -y
sudo apt install openjdk-17-jdk python3-venv -y
sudo apt install python3-poetry -y
sudo apt install python3-pip -y

echo "Ambiente do sistema operacional pronto!"
echo "Agora, certifique-se de ter o Poetry instalado e rode 'poetry install'."