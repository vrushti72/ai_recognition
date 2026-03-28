from insightface.app import FaceAnalysis

print("Downloading model...")

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

print("✅ Model loaded successfully!")