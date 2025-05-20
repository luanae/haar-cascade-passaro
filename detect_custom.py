import cv2
import os

CASCADE_PATH = "cascade/cascade.xml"  # caminho do seu cascade treinado

def detectar_em_imagem(imagem_path):
    img = cv2.imread(imagem_path)
    if img is None:
        print(f"[!] Erro ao carregar: {imagem_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    objetos = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in objetos:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Detecção", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectar_em_webcam():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] Erro ao acessar a webcam.")
        return

    print("[INFO] Pressione 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objetos = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in objetos:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Webcam - Detecção", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detectar_em_diretorio(diretorio):
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    imagens = [f for f in os.listdir(diretorio) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not imagens:
        print("[!] Nenhuma imagem encontrada na pasta.")
        return

    for img_file in imagens:
        caminho = os.path.join(diretorio, img_file)
        print(f"[INFO] Processando: {caminho}")
        detectar_em_imagem(caminho)

def menu():
    print("=== Teste do Classificador Haar Cascade ===")
    print("1. Testar uma imagem")
    print("2. Usar webcam")
    print("3. Testar uma pasta com várias imagens")
    escolha = input("Escolha uma opção (1/2/3): ")

    if escolha == "1":
        caminho = input("Digite o caminho da imagem: ")
        detectar_em_imagem(caminho)
    elif escolha == "2":
        detectar_em_webcam()
    elif escolha == "3":
        pasta = input("Digite o caminho da pasta com imagens: ")
        detectar_em_diretorio(pasta)
    else:
        print("[!] Opção inválida.")

if __name__ == "__main__":
    if not os.path.exists(CASCADE_PATH):
        print("[!] O classificador treinado (cascade.xml) não foi encontrado.")
    else:
        menu()
