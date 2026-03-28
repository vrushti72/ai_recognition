import cv2
import chromadb
import uuid
import os
import time
from datetime import datetime
from insightface.app import FaceAnalysis
import numpy as np

VIDEO_PATH = "C:/Users/Vrushti/OneDrive/Desktop/ai_recognition/videos/video1.mp4"

FACE_MATCH_THRESHOLD = 0.5
SAVE_INTERVAL = 5   # seconds

DB_PATH = "video_face_db"
SAVE_FOLDER = "captured_faces"

# ==============================
# DATABASE
# ==============================

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="face_embeddings")

# ==============================
# AI MODEL
# ==============================

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ==============================
# TRACKING MEMORY
# ==============================

person_counter = {}
person_index_map = {}
last_saved_time = {}
embedding_memory = {}   # 🔥 NEW (for stability)

profile_number = 1

# ==============================
# VIDEO
# ==============================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Video not found")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30

print("🎥 Video started...\n")

# ==============================
# HELPER FUNCTION (cosine similarity)
# ==============================

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()
    if not ret:
        print("\n✅ Video Completed\n")

        print("📊 FINAL REPORT:")
        for pid in person_counter:
            print(f"Profile {person_index_map[pid]} → Seen {person_counter[pid]} times")
        break

    faces = app.get(frame)

    for face in faces:

        if face.det_score < 0.5:
            continue

        x1,y1,x2,y2 = face.bbox.astype(int)

        h, w, _ = frame.shape
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(w,x2)
        y2 = min(h,y2)

        embedding = face.embedding.tolist()
        person_id = None

        # ==========================
        # 🔥 LOCAL TRACKING MATCH (FAST + STABLE)
        # ==========================

        best_match = None
        best_score = -1

        for pid, emb in embedding_memory.items():
            score = cosine_similarity(embedding, emb)

            if score > best_score:
                best_score = score
                best_match = pid

        if best_score > 0.7:   # strong similarity
            person_id = best_match

        # ==========================
        # DATABASE MATCH (fallback)
        # ==========================

        if not person_id and collection.count() > 0:

            results = collection.query(
                query_embeddings=[embedding],
                n_results=1
            )

            if results["distances"]:
                if results["distances"][0][0] < FACE_MATCH_THRESHOLD:
                    person_id = results["metadatas"][0][0]["person_id"]

        # ==========================
        # NEW PERSON
        # ==========================

        if not person_id:
            person_id = f"person_{str(uuid.uuid4())[:6]}"

        # ==========================
        # UPDATE EMBEDDING MEMORY
        # ==========================

        embedding_memory[person_id] = embedding

        # ==========================
        # INIT PERSON
        # ==========================

        if person_id not in person_counter:
            person_counter[person_id] = 0
            person_index_map[person_id] = profile_number
            last_saved_time[person_id] = 0

            print(f"🆕 New Face → Profile {profile_number}")
            profile_number += 1

        # ==========================
        # COUNT
        # ==========================

        person_counter[person_id] += 1

        # ==========================
        # SAVE (controlled)
        # ==========================

        current_time = time.time()

        if current_time - last_saved_time[person_id] > SAVE_INTERVAL:

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size != 0:
                person_folder = os.path.join(SAVE_FOLDER, person_id)
                os.makedirs(person_folder, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(person_folder, f"{timestamp}.jpg")

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

                last_saved_time[person_id] = current_time

        # ==========================
        # DRAW
        # ==========================

        label = f"P{person_index_map[person_id]} ({person_counter[person_id]})"

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("AI Surveillance", frame)

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()