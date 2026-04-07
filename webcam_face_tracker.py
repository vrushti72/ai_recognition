import cv2
import chromadb
import uuid
import os
import time
from datetime import datetime
from insightface.app import FaceAnalysis
PROCESS_INTERVAL = 0.3
FACE_MATCH_THRESHOLD = 0.45

DB_PATH = "webcam_face_db"
SAVE_FOLDER = "captured_faces"

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="face_embeddings",
    metadata={"hnsw:space": "cosine"}
)

print("Database ready")

print("Loading Face Recognition AI...")

# Use default model pack (avoids antelopev2 errors)
app = FaceAnalysis()


app.prepare(ctx_id=0, det_size=(640,640))

print("AI ready")



if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening webcam")
    exit()

print("Webcam started. Press Q to exit.")

last_process_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    if current_time - last_process_time > PROCESS_INTERVAL:

        faces = app.get(frame)

        for face in faces:

            if face.det_score < 0.6:
                continue

            x1,y1,x2,y2 = face.bbox.astype(int)

            # safe bounding box
            h, w, _ = frame.shape
            x1 = max(0,x1)
            y1 = max(0,y1)
            x2 = min(w,x2)
            y2 = min(h,y2)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            embedding = face.embedding.tolist()

            person_id = None

            if collection.count() > 0:

                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=1
                )

                if results["distances"]:

                    distance = results["distances"][0][0]

                    if distance < FACE_MATCH_THRESHOLD:
                        person_id = results["metadatas"][0][0]["person_id"]

            if not person_id:
                person_id = f"person_{str(uuid.uuid4())[:6]}"
                print("New person:", person_id)

   

            person_folder = os.path.join(SAVE_FOLDER, person_id)

            if not os.path.exists(person_folder):
                os.makedirs(person_folder)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"{timestamp}.jpg"
            filepath = os.path.join(person_folder, filename)

            cv2.imwrite(filepath, face_crop)

            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                metadatas=[{
                    "person_id": person_id,
                    "time": timestamp,
                    "image_path": filepath
                }]
            )

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(
                frame,
                person_id,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        last_process_time = current_time

    cv2.imshow("AI Webcam Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
