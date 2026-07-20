import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# Fast model
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(320, 320))

known_names = []
known_embeddings = []

print("Loading known faces...")

# Load known faces
for file in os.listdir("."):
    if file.lower().endswith((".jpg", ".jpeg", ".png")) and file != "test.jpg":

        img = cv2.imread(file)

        if img is None:
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"No face in {file}")
            continue

        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)

        known_embeddings.append(emb)
        known_names.append(os.path.splitext(file)[0])

known_embeddings = np.array(known_embeddings)

print("Loaded:", known_names)

# Test image
img = cv2.imread("test.jpg")

faces = app.get(img)

best_face = None
best_name = "Unknown"
best_score = -1

for face in faces:

    emb = face.embedding
    emb = emb / np.linalg.norm(emb)

    scores = known_embeddings @ emb

    idx = np.argmax(scores)
    score = scores[idx]

    if score > best_score:
        best_score = score
        best_face = face
        best_name = known_names[idx]

# Draw ONLY the best matching face
if best_face is not None and best_score > 0.5:

    x1, y1, x2, y2 = map(int, best_face.bbox)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)

    cv2.putText(
        img,
        f"{best_name} ({best_score:.2f})",
        (x1, y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )
else:
    print("No matching person found.")

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()