import os
import cv2
import numpy as np
from itertools import combinations

DATASET_PATH = "C:/Users/Vrushti/OneDrive/Desktop/ai_recognition/test_data"
THRESHOLD = 0.45

print("🔄 Loading InsightFace model...")

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

print("✅ Model loaded")

# -----------------------------
# LOAD DATASET + EMBEDDINGS
# -----------------------------
data = []

print("\n📂 Reading dataset...")

for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"\n👤 Processing: {person_name}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        print(f"➡️ {img_path}")

        img = cv2.imread(img_path)

        if img is None:
            print("❌ Failed to load image")
            continue

        # -----------------------------
        # TRY NORMAL DETECTION
        # -----------------------------
        faces = app.get(img)

        if len(faces) > 0:
            emb = faces[0].embedding
            print("✅ Face detected (normal pipeline)")

        else:
            # -----------------------------
            # FALLBACK (for low-res cropped images)
            # -----------------------------
            print("⚠️ Detection failed → using fallback")

            img_resized = cv2.resize(img, (112, 112))
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            emb = app.models['recognition'].get_feat(img_resized).flatten()

        # Normalize embedding (VERY IMPORTANT)
        emb = emb / np.linalg.norm(emb)

        data.append((emb, person_name))

print(f"\n📊 Total usable images: {len(data)}")

if len(data) < 2:
    print("❌ Not enough data to compare")
    exit()

# -----------------------------
# COMPARE ALL PAIRS
# -----------------------------
TP, FP, FN = 0, 0, 0
same_distances = []
diff_distances = []

print("\n⚙️ Comparing all pairs...")

for (emb1, label1), (emb2, label2) in combinations(data, 2):

    # Cosine distance (IMPORTANT)
    dist = 1 - np.dot(emb1, emb2)

    same_person = (label1 == label2)
    predicted_same = (dist < THRESHOLD)

    print(f"Distance: {dist:.4f} | Same: {same_person} | Predicted: {predicted_same}")

    if same_person:
        same_distances.append(dist)
    else:
        diff_distances.append(dist)

    if same_person and predicted_same:
        TP += 1
    elif not same_person and predicted_same:
        FP += 1
    elif same_person and not predicted_same:
        FN += 1

# -----------------------------
# METRICS
# -----------------------------
print("\n==============================")
print("📈 FINAL RESULTS")
print("==============================")

print(f"✅ TP (correct match): {TP}")
print(f"❌ FP (wrong match):   {FP}")
print(f"⚠️ FN (missed match):  {FN}")

accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

print(f"\n🎯 Accuracy:  {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🎯 Recall:    {recall:.4f}")

# -----------------------------
# DISTANCE ANALYSIS
# -----------------------------
print("\n==============================")
print("📊 DISTANCE ANALYSIS")
print("==============================")

if same_distances:
    print(f"👤 Same avg: {np.mean(same_distances):.4f}")
    print(f"👤 Same min/max: {np.min(same_distances):.4f} / {np.max(same_distances):.4f}")

if diff_distances:
    print(f"🧍 Different avg: {np.mean(diff_distances):.4f}")
    print(f"🧍 Different min/max: {np.min(diff_distances):.4f} / {np.max(diff_distances):.4f}")
