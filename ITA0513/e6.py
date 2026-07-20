import cv2

cap = cv2.VideoCapture("ex6.mp4")
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))
    fgmask = fgbg.apply(frame)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Pick the largest moving object
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:  # ignore small noise
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Object Capture", frame)

    if cv2.waitKey(10) & 0xFF == ord("1"): break

cap.release()
out.release()
cv2.destroyAllWindows()
