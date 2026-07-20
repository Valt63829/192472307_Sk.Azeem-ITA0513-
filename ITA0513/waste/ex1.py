import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(320, 320))

# Load reference image
reference = cv2.imread("bill.jpg")
ref_faces = app.get(reference)

if len(ref_faces) == 0:
    print("No face found in bill.jpg")
    exit()

# Reference embedding
ref_embedding = ref_faces[0].embedding
ref_embedding /= np.linalg.norm(ref_embedding)

# Load group image
group = cv2.imread("group.jpg")
group_faces = app.get(group)

best_score = -1
best_face = None

# Compare every detected face
for face in group_faces:

    emb = face.embedding
    emb /= np.linalg.norm(emb)

    score = np.dot(ref_embedding, emb)

    if score > best_score:
        best_score = score
        best_face = face

print("Best Similarity:", round(best_score, 3))

# Draw only if similarity is high enough
THRESHOLD = 0.55

if best_face is not None and best_score > THRESHOLD:

    x1, y1, x2, y2 = map(int, best_face.bbox)

    cv2.rectangle(group, (x1, y1), (x2, y2), (0,255,0), 3)

    cv2.putText(
        group,
        f"Match {best_score:.2f}",
        (x1, y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

else:
    print("Person not found.")

cv2.imshow("Result", group)
cv2.waitKey(0)
cv2.destroyAllWindows()