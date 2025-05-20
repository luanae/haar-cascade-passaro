# Haar Cascade Trainer - Detecção de Objetos com OpenCV

Este projeto oferece um pipeline completo para **treinamento, validação e teste de um classificador Haar Cascade personalizado**, utilizando a biblioteca OpenCV. O objetivo é treinar um modelo eficiente para detectar objetos específicos (como pássaros, livros, etc.) com base em imagens positivas anotadas manualmente e imagens negativas.

---

## 📁 Estrutura do Projeto

haar-cascade/
├── dataset/
│ ├── positives/ # Imagens contendo o objeto de interesse
│ └── negatives/ # Imagens sem o objeto
│
├── annotations/ # Arquivos de anotação gerados (positives.txt, negatives.txt)
├── vec/ # Arquivo .vec com as amostras positivas
├── cascade/ # Classificador final treinado (cascade.xml)
├── resultados/ # Imagens testadas e relatório de detecção
│
├── auto_pipeline.py # Pipeline completo para marcação, geração e treino
└── detect_custom.py # Script para testar o classificador treinado

yaml
Copiar
Editar

---

## ✅ Funcionalidades

- Anotação manual interativa das imagens positivas
- Geração automática dos arquivos `positives.txt` e `negatives.txt`
- Criação do arquivo `.vec` com `opencv_createsamples`
- Treinamento do classificador Haar com `opencv_traincascade`
- Teste de detecção em imagens locais, pastas ou URLs
- Geração de relatório com todas as detecções

---

## ⚙️ Requisitos

- Python 3.10+
🧩 Instalação do OpenCV 3.4.11 com Executáveis para Treinamento
Para treinar seu classificador Haar Cascade, é necessário utilizar dois executáveis do OpenCV:

opencv_createsamples.exe

opencv_traincascade.exe

Siga os passos abaixo para instalá-los corretamente:

🔽 1. Baixar o OpenCV 3.4.11
Acesse o link abaixo e baixe o instalador:

📎 Download OpenCV 3.4.11 (vc14_vc15)

📦 2. Extrair para o disco local C:
Após o download:

Execute o instalador opencv-3.4.11-vc14_vc15.exe.

Ele pedirá um local para extração. Escolha:

makefile
Copiar
Editar
C:\opencv\
Isso criará a estrutura:

makefile
Copiar
Editar
C:\opencv\build\x64\vc15\bin\
🛠 3. Adicionar ao PATH do Windows
Para usar os executáveis de qualquer lugar, adicione a pasta bin ao PATH do sistema:

Pressione Win + S, digite "variáveis de ambiente" e abra.

Clique em "Variáveis de Ambiente".

Em "Variáveis do sistema", selecione Path e clique em Editar.

Adicione o seguinte caminho:

makefile
Copiar
Editar
C:\opencv\build\x64\vc15\bin
Clique em OK até fechar todas as janelas.

- Instalar dependências:
  ```bash
  pip install opencv-python
🚀 Como Usar
1. Preparar as imagens
Coloque imagens com o objeto de interesse em dataset/positives/

Coloque imagens sem o objeto em dataset/negatives/

2. Rodar o pipeline de treinamento
bash
Copiar
Editar
python auto_pipeline.py
Isso irá:

Abrir interface de anotação para cada imagem positiva

Validar as anotações

Gerar positives.vec

Treinar o classificador Haar (cascade/cascade.xml)

3. Testar o classificador treinado
bash
Copiar
Editar
python detect_custom.py
Você poderá:

Testar uma imagem (local ou link)

Testar um diretório completo com várias imagens

⚙️ Parâmetros de Treinamento
Tamanho da janela: 50x50

Estágios: 20

Positivos: 90% do total anotado

Negativos: 800 imagens

minHitRate: 0.995

maxFalseAlarmRate: 0.3

⚠️ Tempo estimado de treinamento: até 3 horas, dependendo do hardware e número de imagens.

📊 Relatório
Após os testes, o script detect_custom.py gera automaticamente um relatório em:

bash
Copiar
Editar
resultados/relatorio.txt
O relatório inclui:

Nome da imagem processada

Total de objetos detectados

Coordenadas de cada detecção (x, y, largura, altura)

