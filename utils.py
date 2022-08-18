import numpy as np
import cv2
from mediapipe.python.solutions import pose, drawing_utils, drawing_styles
from matplotlib import pyplot as plt

def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    angle = np.abs(np.rad2deg(np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])))
    if angle > 180:
        angle = 360 - angle
    return angle

def count_curls():

    WIDTH = 1800
    HEIGHT = 1600

    #choose input
    print("Do you want to use a webcam y/n: ", end="")
    is_webcam = input() == "y"

    cap = cv2.VideoCapture(0) if is_webcam else cv2.VideoCapture("assets/video.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    detector = pose
    drawer = drawing_utils


    counter = 0
    stage = None
    labeled = []

    with detector.Pose(
        min_detection_confidence=.5,
        min_tracking_confidence=.5
    ) as poser:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to capture image")
                continue
            # annotated_image = image.copy()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = poser.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # get coordinates
                shoulder = (landmarks[detector.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[detector.PoseLandmark.LEFT_SHOULDER.value].y)
                elbow = (landmarks[detector.PoseLandmark.LEFT_ELBOW.value].x, landmarks[detector.PoseLandmark.LEFT_ELBOW.value].y)
                wrist = (landmarks[detector.PoseLandmark.LEFT_WRIST.value].x, landmarks[detector.PoseLandmark.LEFT_WRIST.value].y)

                # get angle
                angle = get_angle(shoulder, elbow, wrist)

                # count rep
                if angle > 160:
                    stage = "down"
                elif angle < 30 and stage == "down":
                    counter += 1
                    stage = "up"

            except AttributeError:
                if not is_webcam:
                    break
                print("No landmarks found")
                continue


            # show counter
            cv2.rectangle(image, (0, 0), (73, 73), (0, 0, 255), -1)
            cv2.putText(image, str(counter), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


            drawer.draw_landmarks(
                image,
                results.pose_landmarks,
                pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
            )
            im = cv2.resize(image, (WIDTH, HEIGHT))
            cv2.imshow("Face Detection", im)
            labeled.append(im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        output = cv2.VideoWriter("out/output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (WIDTH, HEIGHT))
        for image in labeled:
            output.write(image)
        output.release()

def get_coords(landmarks):
    # get coordinates:
    names = {
        "NOSE": 0,
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13,
        "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15,
        "RIGHT_WRIST": 16,
        "LEFT_PINKY": 17,
        "RIGHT_PINKY": 18,
        "LEFT_INDEX": 19,
        "RIGHT_INDEX": 20,
        "LEFT_THUMB": 21,
        "RIGHT_THUMB": 22,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
        "LEFT_HEEL": 29,
        "RIGHT_HEEL": 30,
        "LEFT_FOOT_INDEX": 31,
        "RIGHT_FOOT_INDEX": 32
    }

    return {landmark: (landmarks[names[landmark]].x, landmarks[names[landmark]].y) for landmark in names}
    
def draw_skeleton(landmarks):
    coords = get_coords(landmarks)
