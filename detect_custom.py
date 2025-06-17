import cv2
import os
import urllib.request
import tempfile
from datetime import datetime

# === Caminho para o classificador Haar Cascade treinado ===
CASCADE_PATH = "cascade/cascade.xml"

# === Diretórios de saída ===
RESULTADOS_PATH = "resultados"
RELATORIO_PATH = os.path.join(RESULTADOS_PATH, "relatorio.txt")
os.makedirs(RESULTADOS_PATH, exist_ok=True)  # Cria a pasta 'resultados' se não existir

# === Inicializa o relatório de detecção ===
with open(RELATORIO_PATH, "w") as rel:
    rel.write("=== RELATÓRIO DE DETECÇÃO ===\n\n")


# === Função: registrar as detecções feitas em uma imagem no relatório ===
def registrar_no_relatorio(nome_img, objetos):
    with open(RELATORIO_PATH, "a") as rel:
        rel.write(f"Imagem: {nome_img}\n")
        rel.write(f"Total detectado(s): {len(objetos)}\n")
        for idx, (x, y, w, h) in enumerate(objetos, 1):
            rel.write(f"  [{idx}] x={x}, y={y}, w={w}, h={h}\n")
        rel.write("\n")


# === Função: detectar objetos (pássaros) em uma única imagem local ou de URL ===
def detectar_em_imagem(origem, nome_saida=None):
    temp_file = None  # Para armazenar temporariamente a imagem se for de URL

    # Caso a origem seja um link, baixa a imagem para uso local
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

    # Lê a imagem local
    img = cv2.imread(origem)
    if img is None:
        print(f"[!] Imagem inválida: {origem}")
        return

    # Converte a imagem para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Carrega o classificador Haar Cascade
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    # Aplica a detecção
    objetos = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Desenha os retângulos sobre os objetos detectados
    for (x, y, w, h) in objetos:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Define o nome do arquivo de saída
    if not nome_saida:
        nome_saida = os.path.basename(origem).split("?")[0]
    saida_path = os.path.join(RESULTADOS_PATH, f"resultado_{nome_saida}")
    cv2.imwrite(saida_path, img)

    # Exibe a imagem com as detecções
    cv2.imshow("Detecção", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Registra no relatório e remove temporário (se houver)
    registrar_no_relatorio(nome_saida, objetos)
    if temp_file:
        os.remove(temp_file.name)

    print(f"[✓] Resultado salvo: {saida_path}")


# === Função: detectar objetos em todas as imagens de uma pasta ===
def detectar_em_diretorio(diretorio):
    imagens = [f for f in os.listdir(diretorio) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not imagens:
        print("[!] Nenhuma imagem encontrada na pasta.")
        return

    for img_file in imagens:
        caminho = os.path.join(diretorio, img_file)
        print(f"[INFO] Processando: {caminho}")
        detectar_em_imagem(caminho, nome_saida=img_file)


# === Função: detectar objetos em tempo real pela webcam e salvar frame ===
def detectar_em_tempo_real():
    cap = cv2.VideoCapture(0)  # Ativa a webcam padrão
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    total_detectados = 0
    objetos_passados = []  # Armazena centros de detecções anteriores

    print("[INFO] Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Erro ao acessar a webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objetos = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        novas_detectadas = []
        for (x, y, w, h) in objetos:
            centro = (x + w // 2, y + h // 2)
            novas_detectadas.append(centro)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        salvar_frame = False

        # Verifica se há uma nova detecção distinta
        for centro in novas_detectadas:
            if all(abs(centro[0] - antigo[0]) > 30 or abs(centro[1] - antigo[1]) > 30 for antigo in objetos_passados):
                total_detectados += 1
                objetos_passados.append(centro)
                salvar_frame = True

        # Mostra contador na tela
        cv2.putText(frame, f"Passaros detectados: {total_detectados}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Salva frame com detecção nova
        if salvar_frame:
            nome_arquivo = f"webcam_deteccao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            caminho_arquivo = os.path.join(RESULTADOS_PATH, nome_arquivo)
            cv2.imwrite(caminho_arquivo, frame)
            registrar_no_relatorio(nome_arquivo, objetos)
            print(f"[✓] Frame salvo: {caminho_arquivo}")

        # Exibe o vídeo em tempo real
        cv2.imshow("Detecção ao Vivo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[✓] Detecção encerrada. Total de pássaros detectados: {total_detectados}")


# === Função de menu para interação do usuário ===
def menu():
    print("=== Teste do Classificador Haar Cascade ===")
    print("1. Testar uma imagem (local ou link)")
    print("2. Testar uma pasta com várias imagens")
    print("3. Detectar pássaros pela webcam (ao vivo)")
    escolha = input("Escolha uma opção (1/2/3): ")

    if escolha == "1":
        origem = input("Digite o caminho da imagem ou URL: ")
        detectar_em_imagem(origem)
    elif escolha == "2":
        pasta = input("Digite o caminho da pasta com imagens: ")
        if os.path.exists(pasta):
            detectar_em_diretorio(pasta)
        else:
            print("[!] Pasta não encontrada.")
    elif escolha == "3":
        detectar_em_tempo_real()
    else:
        print("[!] Opção inválida.")


# === Ponto de entrada principal do script ===
if __name__ == "__main__":
    if not os.path.exists(CASCADE_PATH):
        print("[!] O classificador treinado (cascade.xml) não foi encontrado.")
    else:
        menu()
