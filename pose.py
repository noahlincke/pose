import imutils
from imutils import face_utils
from imutils.video import VideoStream
import time
import dlib
import cv2
import numpy as np

shape_model = "shape_predictor_68.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_model)

# CHANGE FRAMERATE ACCORDING TO WEBCAM
vs = VideoStream(src=0, framerate=30).start()

time.sleep(2.0)
frame = vs.read()
frame_scale = 1

# Store output to 'poseEstimate.mp4'.
out = cv2.VideoWriter(
    "poseEstimate.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    15,
    (frame.shape[1], frame.shape[0]),
)

# The 6 main facial landmarks (initialized to 0)
# Used for pose estimation through solving
# perspective n-points (n=6)
image_points = np.array(
    [
        (0, 0),  # Nose tip 34
        (0, 0),  # Chin 9
        (0, 0),  # Left eye left corner 37
        (0, 0),  # Right eye right corner 46
        (0, 0),  # Left mouth corner 49
        (0, 0),  # Right mouth corner 55
    ],
    dtype="double",
)

# The positions of the 6 facial landmarks on an
# average face in 3D space
# Used for pose estimation through solving
# perspective n-points (n=6)
model_points = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip 34
        (0.0, -330.0, -65.0),  # Chin 9
        (-225.0, 170.0, -135.0),  # Left eye left corner 37
        (225.0, 170.0, -135.0),  # Right eye right corner 46
        (-150.0, -150.0, -125.0),  # Left Mouth corner 49
        (150.0, -150.0, -125.0),  # Right mouth corner 55
    ]
)

# Draws point on frame with label "i+1" and specified color
def drawPoint(frame, i, x, y, color):
    cv2.circle(frame, (x, y), 1, color, -1)
    cv2.putText(
        frame,
        str(i + 1),
        (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
    )


# Draws each landmark given by shape onto the frame
# The 6 key landmarks are drawn in green, while the rest are red
def drawUpdateLandmarks(frame, shape):
    for (i, (x, y)) in enumerate(shape):
        if i == 33:  # Nose tip
            image_points[0] = np.array([x, y], dtype="double")
            drawPoint(frame, i, x, y, (0, 255, 0))
        elif i == 8:  # Chin
            image_points[1] = np.array([x, y], dtype="double")
            drawPoint(frame, i, x, y, (0, 255, 0))
        elif i == 36:  # Left eye left corner
            image_points[2] = np.array([x, y], dtype="double")
            drawPoint(frame, i, x, y, (0, 255, 0))
        elif i == 45:  # Right eye right corner
            image_points[3] = np.array([x, y], dtype="double")
            drawPoint(frame, i, x, y, (0, 255, 0))
        elif i == 48:  # Left mouth corner
            image_points[4] = np.array([x, y], dtype="double")
            drawPoint(frame, i, x, y, (0, 255, 0))
        elif i == 54:  # Right mouth corner
            image_points[5] = np.array([x, y], dtype="double")
            drawPoint(frame, i, x, y, (0, 255, 0))
        else:  # All other 62 landmarks
            drawPoint(frame, i, x, y, (0, 0, 255))


# Moves the largest rectangle given by the facial detector
# (face in focus) to the front of the list
def sortRects(rects):
    maxSize = 0
    for i in range(len(rects)):
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rects[i])
        if bW * bH > maxSize:
            rects.insert(0, rects.pop(i))
            maxSize = bW * bH
    return rects


# Continuously read frame the source, perform detections and
# drawings, and view and write the frame to the output destination
# Stop when "q" is pressed
fps = []
while True:
    start = time.time()
    frame = vs.read()
    if frame_scale != 1:
        frame = imutils.resize(
            frame,
            round(frame.shape[1] * frame_scale),
            round(frame.shape[0] * frame_scale),
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    # Get box detections and order based on size descending
    rects = detector(gray, 0)
    rects = sortRects(rects)
    for i, rect in enumerate(rects):
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        if i == 0:  # Largest (focus) face gets a green bounding box
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        else:
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 0, 255), 1)

        # Get facial landmark predictions and convert to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        drawUpdateLandmarks(frame, shape)

        # Pose Estimation
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # Project a 3D point (0,0,1000) into 2D image using the calculated
        # perspective change
        (nose_end_point2D, jacobian) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]),
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs,
        )

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        # Draw line starting from actual nose tip (p1) to projected endpoint (p2)
        cv2.line(frame, p1, p2, (255, 0, 0), 2)

    end = time.time()
    fps.append(1 / (end - start))
    text = "FPS: {}".format(round(fps[-1], 2))
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Video", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("Average FPS: {}".format(round(np.mean(fps), 2)))
cv2.destroyAllWindows()
vs.stop()
