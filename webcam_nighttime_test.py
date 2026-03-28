import cv2
import chromadb
import threading
import time
import uuid
import numpy as np
from insightface.app import FaceAnalysis
from datetime import datetime
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================
WEBCAM_INDEX = 0
PROCESS_INTERVAL = 0.1
FACE_MATCH_THRESHOLD = 0.55

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "live_video_db")
SAVE_FOLDER = os.path.join(SCRIPT_DIR, "captured_faces")

# ==========================================
# DATABASE
# ==========================================
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_or_create_collection(
    name="face_embeddings",
    metadata={"hnsw:space": "cosine"}
)

# ==========================================
# MODEL (SAFE CPU VERSION)
# ==========================================
print("⏳ Loading InsightFace (CPU mode for safety)...")

app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

print("✅ Model loaded successfully")

# ==========================================
# WEBCAM STREAM
# ==========================================
class WebcamStream:
    def __init__(self, index):
        self.stream = cv2.VideoCapture(index)
        self.stopped = False
        self.frame = None
        self.last_process_time = time.time()
        self.printed_shape = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                continue
            self.frame = frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# PREPROCESSING (NIGHT PIPELINE)
# ==========================================
def preprocess_frame(frame, cam):

    # Print frame shape once
    if not cam.printed_shape:
        print("📏 Frame shape:", frame.shape)
        cam.printed_shape = True

    # Convert grayscale → BGR if needed
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # 1. Contrast
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20)

    # 2. Denoise
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # 3. Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)

    # 4. CLAHE
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0])
    frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return frame

# ==========================================
# PROCESS FUNCTION
# ==========================================
def process_frame(cam):
    if cam.frame is None:
        return

    current_time = time.time()
    if current_time - cam.last_process_time < PROCESS_INTERVAL:
        return

    frame = cam.frame
    processed = preprocess_frame(frame, cam)

    faces = app.get(processed)

    if faces:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for face in faces:
            if face.det_score < 0.5:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int)
            face_crop = processed[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            embedding = face.embedding.tolist()
            person_id = None

            # DB search
            if collection.count() > 0:
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=1
                )

                if results['distances'] and len(results['distances'][0]) > 0:
                    distance = results['distances'][0][0]
                    if distance < FACE_MATCH_THRESHOLD:
                        person_id = results['metadatas'][0][0].get("person_id")

            if not person_id:
                person_id = f"person_{str(uuid.uuid4())[:8]}"
                print(f"🆕 New Person: {person_id}")

            # Save image
            person_folder = os.path.join(SAVE_FOLDER, person_id)
            os.makedirs(person_folder, exist_ok=True)

            file_path = os.path.join(person_folder, f"{timestamp}.jpg")
            cv2.imwrite(file_path, face_crop)

            # Save to DB
            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                metadatas={
                    "person_id": person_id,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": file_path
                }
            )

    cam.last_process_time = current_time

# ==========================================
# MAIN LOOP
# ==========================================
cam = WebcamStream(WEBCAM_INDEX).start()

print("🚀 Webcam running... Press 'q' to exit")

try:
    while True:
        process_frame(cam)

        if cam.frame is not None:
            cv2.imshow("Webcam", cam.frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cam.stop()
cv2.destroyAllWindows()
print("🛑 System stopped")