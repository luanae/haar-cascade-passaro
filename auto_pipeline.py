import cv2
import os
import subprocess

# ==== Caminhos principais ====
POSITIVE_PATH = "dataset/positives"  # Imagens com o objeto (positivas)
NEGATIVE_PATH = "dataset/negatives"  # Imagens sem o objeto (negativas)
ANNOTATIONS_PATH = "annotations"  # Onde serão salvas as anotações manuais
VEC_DIR = "vec"  # Pasta onde será salvo o arquivo .vec
CASCADE_DIR = "cascade"  # Pasta onde será salvo o modelo treinado

# ==== Arquivos gerados ====
POSITIVES_FILE = os.path.join(ANNOTATIONS_PATH, "positives.txt")
NEGATIVES_FILE = os.path.join(ANNOTATIONS_PATH, "negatives.txt")
VEC_FILE = os.path.join(VEC_DIR, "positives.vec")
CASCADE_XML = os.path.join(CASCADE_DIR, "cascade.xml")

# ==== Caminho dos executáveis OpenCV ====
CREATESAMPLES_PATH = r"C:\opencv\build\x64\vc15\bin\opencv_createsamples.exe"
TRAINCASCADE_PATH = r"C:\opencv\build\x64\vc15\bin\opencv_traincascade.exe"

# ==== Variáveis globais de anotação ====
drawing = False
ix, iy = -1, -1
bboxes = []
img_copy = None

# ==== Função para desenhar múltiplos retângulos na imagem ====
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img_copy, bboxes, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = img.copy()
        for box in bboxes:
            cv2.rectangle(img_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.imshow("Selecione os objetos", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x0, y0 = min(ix, x), min(iy, y)
        w, h = abs(ix - x), abs(iy - y)
        if w > 0 and h > 0:
            bboxes.append([x0, y0, w, h])
            img_copy = img.copy()
            for box in bboxes:
                cv2.rectangle(img_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
            cv2.imshow("Selecione os objetos", img_copy)

# ==== Função para marcar as regiões nas imagens positivas ====
def marcar_imagens():
    os.makedirs(ANNOTATIONS_PATH, exist_ok=True)
    imagens = sorted([f for f in os.listdir(POSITIVE_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    with open(POSITIVES_FILE, "w") as f:
        for nome_arquivo in imagens:
            global img, img_copy, bboxes
            bboxes = []
            caminho = os.path.join(POSITIVE_PATH, nome_arquivo)
            img = cv2.imread(caminho)
            if img is None:
                print(f"[!] Erro ao abrir: {caminho}")
                continue

            altura, largura = img.shape[:2]
            img_copy = img.copy()
            cv2.namedWindow("Selecione os objetos")
            cv2.setMouseCallback("Selecione os objetos", draw_rectangle)
            cv2.imshow("Selecione os objetos", img_copy)

            print(f"🔍 Marque os objetos na imagem: {nome_arquivo}. [ENTER] confirma | [BACKSPACE] desfaz última | [ESC] pula imagem")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 13:  # ENTER
                    if bboxes and any(w > 0 and h > 0 for _, _, w, h in bboxes):
                        full_path = os.path.abspath(caminho).replace("\\", "/")
                        coordenadas_validas = []
                        for x, y, w, h in bboxes:
                            if (x + w <= largura) and (y + h <= altura):
                                coordenadas_validas.extend([x, y, w, h])
                            else:
                                print(f"[✗] BBox fora dos limites. Ignorada: {(x, y, w, h)}")

                        if coordenadas_validas:
                            num_validas = len(coordenadas_validas) // 4
                            linha = f"{full_path} {num_validas}"
                            for i in range(num_validas):
                                linha += f" {coordenadas_validas[i * 4]} {coordenadas_validas[i * 4 + 1]} {coordenadas_validas[i * 4 + 2]} {coordenadas_validas[i * 4 + 3]}"
                            f.write(linha + "\n")
                            print(f"[✓] {num_validas} objeto(s) anotado(s).")
                        else:
                            print("[!] Nenhum bbox válido. Pulando imagem.")
                    else:
                        print("[!] Nenhum objeto marcado. Pulando imagem.")
                    break
                elif key == 8 and bboxes:  # BACKSPACE
                    removido = bboxes.pop()
                    print(f"[↩] Removido: {removido}")
                    img_copy = img.copy()
                    for box in bboxes:
                        cv2.rectangle(img_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
                    cv2.imshow("Selecione os objetos", img_copy)
                elif key == 27:  # ESC
                    print("[!] Imagem ignorada.")
                    break

            cv2.destroyAllWindows()

    print("[✓] Todas as anotações foram salvas em:", POSITIVES_FILE)

# ==== As demais funções continuam iguais ====
def gerar_negatives_txt():
    os.makedirs(ANNOTATIONS_PATH, exist_ok=True)
    with open(NEGATIVES_FILE, 'w') as f:
        for filename in os.listdir(NEGATIVE_PATH):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.abspath(os.path.join(NEGATIVE_PATH, filename)).replace("\\", "/")
                f.write(f"{full_path}\n")
    print("[✓] Arquivo negatives.txt gerado.")

def contar_amostras():
    with open(POSITIVES_FILE, "r") as f:
        return sum(1 for line in f if line.strip())

def validar_anotacoes():
    print("[INFO] Validando anotações...")
    erros = 0
    with open(POSITIVES_FILE, "r") as f:
        for linha in f:
            partes = linha.strip().split()
            if len(partes) < 6 or (len(partes) - 2) % 4 != 0:
                print("[!] Formato inválido:", linha)
                erros += 1
                continue

            img_path = partes[0]
            try:
                num_objs = int(partes[1])
                bboxes = list(map(int, partes[2:]))
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[!] Imagem não encontrada: {img_path}")
                    erros += 1
                    continue

                altura, largura = img.shape[:2]
                for i in range(num_objs):
                    x, y, w, h = bboxes[i*4:i*4+4]
                    if x + w > largura or y + h > altura:
                        print(f"[✗] BBox fora dos limites: {img_path} ({x},{y},{w},{h})")
                        erros += 1
            except:
                print("[!] Erro ao processar linha:", linha)
                erros += 1

    if erros == 0:
        print("[✓] Todas as anotações são válidas.")
    else:
        print(f"[!] {erros} erro(s) encontrados. Corrija antes de continuar.")
        exit()

def gerar_vec():
    os.makedirs(VEC_DIR, exist_ok=True)
    total_amostras = contar_amostras()
    if total_amostras < 5:
        print(f"[!] Mínimo de 5 amostras exigido. Encontradas: {total_amostras}")
        exit()

    info_abspath = os.path.abspath(POSITIVES_FILE)
    vec_abspath = os.path.abspath(VEC_FILE)
    annotations_dir = os.path.dirname(info_abspath)
    info_filename = os.path.basename(info_abspath)

    cmd = [
        CREATESAMPLES_PATH,
        "-info", info_filename,
        "-num", str(total_amostras),
        "-w", "50",
        "-h", "50",
        "-vec", vec_abspath
    ]

    print("Executando:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=annotations_dir)

    if result.returncode != 0:
        print("[✗] Falha ao gerar .vec")
        exit()

    print(f"[✓] Arquivo .vec criado com sucesso com {total_amostras} amostras.")
    return total_amostras

def treinar_cascade(num_amostras_validas):
    num_pos = max(1, num_amostras_validas - 1)

    os.makedirs(CASCADE_DIR, exist_ok=True)
    bg_path = os.path.abspath(NEGATIVES_FILE).replace("\\", "/")
    vec_path = os.path.abspath(VEC_FILE).replace("\\", "/")

    cmd = [
        TRAINCASCADE_PATH,
        "-data", CASCADE_DIR,
        "-vec", vec_path,
        "-bg", bg_path,
        "-numPos", str(num_pos),
        "-numNeg", "521",
        "-numStages", "15",
        "-w", "50",
        "-h", "50",
        "-maxFalseAlarmRate", "0.2",
        "-minHitRate", "0.995"
    ]

    print("Executando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[✓] Modelo Haar Cascade treinado com sucesso.")

# ==== Execução principal ====
if __name__ == "__main__":
    marcar_imagens()
    gerar_negatives_txt()
    validar_anotacoes()
    total_valido = gerar_vec()
    treinar_cascade(total_valido)
