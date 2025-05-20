import cv2
import os
import subprocess

# Caminhos principais
POSITIVE_PATH = "dataset/positives"
NEGATIVE_PATH = "dataset/negatives"
ANNOTATIONS_PATH = "annotations"
VEC_DIR = "vec"
CASCADE_DIR = "cascade"

POSITIVES_FILE = os.path.join(ANNOTATIONS_PATH, "positives.txt")
NEGATIVES_FILE = os.path.join(ANNOTATIONS_PATH, "negatives.txt")
VEC_FILE = os.path.join(VEC_DIR, "positives.vec")
CASCADE_XML = os.path.join(CASCADE_DIR, "cascade.xml")

# Executáveis OpenCV
CREATESAMPLES_PATH = r"C:\opencv\build\x64\vc15\bin\opencv_createsamples.exe"
TRAINCASCADE_PATH = r"C:\opencv\build\x64\vc15\bin\opencv_traincascade.exe"

# Variáveis globais de marcação
drawing = False
ix, iy = -1, -1
bbox = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bbox, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = img.copy()
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Selecione o objeto", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x0, y0 = min(ix, x), min(iy, y)
        w, h = abs(ix - x), abs(iy - y)
        if w > 0 and h > 0:
            bbox = [x0, y0, w, h]
            cv2.rectangle(img_copy, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
            cv2.imshow("Selecione o objeto", img_copy)

def marcar_imagens():
    os.makedirs(ANNOTATIONS_PATH, exist_ok=True)
    imagens = sorted([f for f in os.listdir(POSITIVE_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    with open(POSITIVES_FILE, "w") as f:
        for nome_arquivo in imagens:
            global img, img_copy, bbox
            bbox = []
            caminho = os.path.join(POSITIVE_PATH, nome_arquivo)
            img = cv2.imread(caminho)
            if img is None:
                print(f"[!] Erro ao abrir: {caminho}")
                continue

            altura, largura = img.shape[:2]

            img_copy = img.copy()
            cv2.namedWindow("Selecione o objeto")
            cv2.setMouseCallback("Selecione o objeto", draw_rectangle)
            cv2.imshow("Selecione o objeto", img)

            print(f"🔍 Marque o objeto em: {nome_arquivo}. Pressione [ENTER] para confirmar ou [ESC] para pular.")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 13 and bbox and bbox[2] > 0 and bbox[3] > 0:
                    x, y, w, h = bbox
                    if (x + w <= largura) and (y + h <= altura):
                        full_path = os.path.abspath(caminho).replace("\\", "/")
                        f.write(f"{full_path} 1 {x} {y} {w} {h}\n")
                        print(f"[✓] Salvo: {bbox}")
                    else:
                        print(f"[✗] Bounding box fora dos limites. Ignorada: {bbox}")
                    break
                elif key == 27:
                    print("[!] Imagem ignorada.")
                    break

            cv2.destroyAllWindows()

    print("[✓] Todas as anotações foram salvas em:", POSITIVES_FILE)

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
            if len(partes) != 6:
                print("[!] Formato inválido:", linha)
                erros += 1
                continue

            img_path, _, x, y, w, h = partes
            x, y, w, h = map(int, (x, y, w, h))
            img = cv2.imread(img_path)
            if img is None:
                print(f"[!] Imagem não encontrada: {img_path}")
                erros += 1
                continue

            altura, largura = img.shape[:2]
            if x + w > largura or y + h > altura:
                print(f"[✗] BBox fora dos limites: {img_path} ({x},{y},{w},{h})")
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

    # Criação do .vec com o MESMO número de amostras do .txt
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
        "-numNeg", "50",
        "-numStages", "10",
        "-w", "50",
        "-h", "50",
        "-maxFalseAlarmRate", "0.5",
        "-minHitRate", "0.995"
    ]

    print("Executando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[✓] Modelo Haar Cascade treinado com sucesso.")


# Execução principal
if __name__ == "__main__":
    marcar_imagens()
    gerar_negatives_txt()
    validar_anotacoes()
    total_valido = gerar_vec()
    treinar_cascade(total_valido)

