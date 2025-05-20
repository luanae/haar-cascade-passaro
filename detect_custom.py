import cv2
import os
import urllib.request
import tempfile

# Caminho do classificador Haar Cascade treinado (gerado após treinamento)
CASCADE_PATH = "cascade/cascade.xml"

# Pasta onde as imagens resultantes e o relatório serão salvos
RESULTADOS_PATH = "resultados"
RELATORIO_PATH = os.path.join(RESULTADOS_PATH, "relatorio.txt")
os.makedirs(RESULTADOS_PATH, exist_ok=True)

# Criação ou reinicialização do arquivo de relatório
with open(RELATORIO_PATH, "w") as rel:
    rel.write("=== RELATÓRIO DE DETECÇÃO ===\n\n")

# Função que grava no relatório as detecções feitas em cada imagem
def registrar_no_relatorio(nome_img, objetos):
    with open(RELATORIO_PATH, "a") as rel:
        rel.write(f"Imagem: {nome_img}\n")
        rel.write(f"Total detectado(s): {len(objetos)}\n")
        for idx, (x, y, w, h) in enumerate(objetos, 1):
            rel.write(f"  [{idx}] x={x}, y={y}, w={w}, h={h}\n")
        rel.write("\n")

# Função para detectar objetos em uma imagem (local ou por URL)
def detectar_em_imagem(origem, nome_saida=None):
    temp_file = None

    # Se for uma URL, baixa a imagem e salva temporariamente
    if origem.startswith("http://") or origem.startswith("https://"):
        print("[INFO] Baixando imagem da web...")
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            urllib.request.urlretrieve(origem, temp_file.name)
            origem = temp_file.name
            print(f"[✓] Imagem salva temporariamente: {origem}")
        except Exception as e:
            print(f"[!] Erro ao baixar imagem: {e}")
            return

    # Leitura da imagem
    img = cv2.imread(origem)
    if img is None:
        print(f"[!] Imagem inválida: {origem}")
        return

    # Conversão para escala de cinza (requisito do classificador Haar)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Carrega o classificador Haar Cascade treinado
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    # Detecta objetos na imagem com parâmetros ajustáveis
    objetos = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Desenha retângulos verdes sobre as detecções
    for (x, y, w, h) in objetos:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Define nome do arquivo de saída
    if not nome_saida:
        nome_saida = os.path.basename(origem).split("?")[0]
    saida_path = os.path.join(RESULTADOS_PATH, f"resultado_{nome_saida}")
    cv2.imwrite(saida_path, img)

    # Exibe imagem com resultados
    cv2.imshow("Detecção", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Grava informações no relatório
    registrar_no_relatorio(nome_saida, objetos)

    # Remove imagem temporária baixada (se houver)
    if temp_file:
        os.remove(temp_file.name)

    print(f"[✓] Resultado salvo: {saida_path}")

# Função para detectar objetos em várias imagens de uma pasta
def detectar_em_diretorio(diretorio):
    # Filtra imagens válidas na pasta
    imagens = [f for f in os.listdir(diretorio) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not imagens:
        print("[!] Nenhuma imagem encontrada na pasta.")
        return

    # Processa imagem por imagem
    for img_file in imagens:
        caminho = os.path.join(diretorio, img_file)
        print(f"[INFO] Processando: {caminho}")
        detectar_em_imagem(caminho, nome_saida=img_file)

# Menu interativo para o usuário escolher o modo de uso
def menu():
    print("=== Teste do Classificador Haar Cascade ===")
    print("1. Testar uma imagem (local ou link)")
    print("2. Testar uma pasta com várias imagens")
    escolha = input("Escolha uma opção (1/2): ")

    if escolha == "1":
        origem = input("Digite o caminho da imagem ou URL: ")
        detectar_em_imagem(origem)
    elif escolha == "2":
        pasta = input("Digite o caminho da pasta com imagens: ")
        if os.path.exists(pasta):
            detectar_em_diretorio(pasta)
        else:
            print("[!] Pasta não encontrada.")
    else:
        print("[!] Opção inválida.")

# Ponto de entrada principal
if __name__ == "__main__":
    # Verifica se o classificador foi treinado e está disponível
    if not os.path.exists(CASCADE_PATH):
        print("[!] O classificador treinado (cascade.xml) não foi encontrado.")
    else:
        menu()
