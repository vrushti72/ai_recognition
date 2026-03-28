import streamlit as st
import chromadb
import pandas as pd
from PIL import Image
import os
import time

# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(page_title="Face Surveillance Dashboard", layout="wide")
st.title("🎥 Video Face Surveillance Dashboard")

# ==============================
# DATABASE PATH (UPDATED)
# ==============================

DB_PATH = "video_face_db"

# ==============================
# CONNECT DATABASE
# ==============================

if not os.path.exists(DB_PATH):
    st.error("❌ video_face_db not found. Run video script first.")
    st.stop()

client = chromadb.PersistentClient(path=DB_PATH)

try:
    collection = client.get_collection("face_embeddings")
except:
    st.error("❌ Collection not found in video DB.")
    st.stop()

data = collection.get()

# ==============================
# CHECK DATA
# ==============================

if not data or not data.get("metadatas"):
    st.warning("⚠️ No data found in video DB.")
    st.stop()

df = pd.DataFrame(data["metadatas"])

# clean bad rows
df = df.dropna(subset=["person_id", "image_path"])

# sort latest first
if "time" in df.columns:
    df = df.sort_values(by="time", ascending=False)

# ==============================
# SIDEBAR FILTER
# ==============================

st.sidebar.header("🔎 Filters")

people = df["person_id"].unique().tolist()

selected_person = st.sidebar.selectbox(
    "Select Person",
    ["All"] + people
)

# ==============================
# FILTER
# ==============================

if selected_person != "All":
    df = df[df["person_id"] == selected_person]

# ==============================
# STATS
# ==============================

col1, col2 = st.columns(2)

col1.metric("Total Records", len(df))
col2.metric("Unique People", len(people))

st.divider()

# ==============================
# DISPLAY
# ==============================

st.subheader("📸 Detected Faces")

for _, row in df.iterrows():

    col1, col2 = st.columns([1, 3])

    img_path = row.get("image_path", "")

    with col1:
        try:
            if os.path.exists(img_path):
                st.image(Image.open(img_path), width=150)
            else:
                st.warning("Image missing")
        except:
            st.warning("Corrupt image")

    with col2:
        st.markdown(f"**Person ID:** `{row['person_id']}`")
        st.markdown(f"**Time:** {row.get('time', 'N/A')}")

    st.divider()

# ==============================
# AUTO REFRESH
# ==============================

time.sleep(3)
st.rerun()