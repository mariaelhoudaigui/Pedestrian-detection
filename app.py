import os
from flask import Flask, render_template, request, send_from_directory, url_for
from ultralytics import YOLO
import cv2
from datetime import datetime

# Création de l'application Flask
app = Flask(__name__)

# Chemin du dossier pour stocker les fichiers uploadés
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Créer le dossier s'il n'existe pas

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Charger le modèle YOLOv8
model = YOLO("best.pt")

# Fonction pour traiter une image
def process_image(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    annotated_image = results[0].plot(labels=False)

    # Sauvegarder l'image annotée dans le dossier static/uploads
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_image.jpg')
    cv2.imwrite(output_image_path, annotated_image)

    return 'annotated_image.jpg'

# Fonction pour traiter une vidéo
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    annotated_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec H.264 pour compatibilité navigateur
    out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot(labels=False)
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if os.path.exists(annotated_video_path):
        print(f"Annotated video saved: {annotated_video_path}")
    else:
        print("Error: Video not saved properly.")

    return 'annotated_video.mp4'

# Route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Route pour télécharger une image
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image part'
    image_file = request.files['image']
    if image_file.filename == '':
        return 'No selected file'
    if image_file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        # Traitement de l'image avec YOLO et sauvegarde de l'image annotée
        annotated_image_filename = process_image(image_path)

        return render_template('index.html', image_path=image_file.filename,
                               annotated_image=annotated_image_filename)

# Route pour télécharger une vidéo
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No video part'
    video_file = request.files['video']
    if video_file.filename == '':
        return 'No selected file'
    if video_file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)

        annotated_video_filename = process_video(video_path)

        return render_template('index.html', video_path=video_file.filename,
                               annotated_video=annotated_video_filename, timestamp=datetime.now().timestamp())

# Route pour afficher les fichiers sauvegardés (image ou vidéo)
@app.route('/static/uploads/<path:filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
