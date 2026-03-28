import cv2
import chromadb
import uuid
import os
import time
from datetime import datetime
from insightface.app import FaceAnalysis

# ==============================
# CONFIG
# ==============================

VIDEO_PATH = "C:/Users/Vrushti/OneDrive/Desktop/ai_recognition/videos/video1.mp4"
PROCESS_INTERVAL = 0.3
FACE_MATCH_THRESHOLD = 0.45

DB_PATH = "video_face_db"
SAVE_FOLDER = "captured_faces"

# ==============================
# DATABASE
# ==============================

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="face_embeddings",
    metadata={"hnsw:space": "cosine"}
)

# ==============================
# AI MODEL
# ==============================

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# ==============================
# STORAGE
# ==============================

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# 🔥 TRACKING DICTIONARY
person_counter = {}
person_index_map = {}   # person_id → profile number
profile_number = 1

# ==============================
# VIDEO
# ==============================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Video not found")
    exit()

print("🎥 Video started...\n")

last_process_time = 0

# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("\n✅ Video Completed")
        print("\n📊 FINAL REPORT:")
        
        for pid in person_counter:
            print(f"Profile {person_index_map[pid]} seen {person_counter[pid]} times")
        
        break

    current_time = time.time()

    if current_time - last_process_time > PROCESS_INTERVAL:

        faces = app.get(frame)

        for face in faces:

            if face.det_score < 0.6:
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
            # SEARCH DATABASE
            # ==========================

            if collection.count() > 0:

                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=1
                )

                if results["distances"]:
                    distance = results["distances"][0][0]

                    if distance < FACE_MATCH_THRESHOLD:
                        person_id = results["metadatas"][0][0]["person_id"]

            # ==========================
            # NEW PERSON
            # ==========================

            if not person_id:
                person_id = f"person_{str(uuid.uuid4())[:6]}"
                
                person_index_map[person_id] = profile_number
                person_counter[person_id] = 0

                print(f"\n🆕 New Face Detected → Profile {profile_number}")
                profile_number += 1

            # ==========================
            # COUNT INCREMENT
            # ==========================

            person_counter[person_id] += 1

            print(f"👁 Seen → Profile {person_index_map[person_id]} | Count = {person_counter[person_id]}")

            # ==========================
            # SAVE IMAGE
            # ==========================

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size != 0:
                person_folder = os.path.join(SAVE_FOLDER, person_id)
                os.makedirs(person_folder, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(person_folder, f"{timestamp}.jpg")
                cv2.imwrite(filepath, face_crop)

            # ==========================
            # STORE IN DB
            # ==========================

            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                metadatas=[{
                    "person_id": person_id,
                    "time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "image_path": filepath
                }]
            )

            # ==========================
            # DRAW
            # ==========================

            label = f"P{person_index_map[person_id]} ({person_counter[person_id]})"

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        last_process_time = current_time

    cv2.imshow("AI Surveillance", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()