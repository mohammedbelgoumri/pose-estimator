import cv2
import mediapipe as mp
from utils import get_angle

WIDTH = 1800
HEIGHT = 1600

def main():

    #choose input
    print("Do you want to use a webcam y/n: ", end="")
    is_webcam = input() == "y"

    cap = cv2.VideoCapture(0) if is_webcam else cv2.VideoCapture("assets/video.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    detector = mp.solutions.pose
    drawer = mp.solutions.drawing_utils


    counter = 0
    stage = None
    labeled = []

    with detector.Pose(
        min_detection_confidence=.5,
        min_tracking_confidence=.5
    ) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to capture image")
                continue
            # annotated_image = image.copy()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

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
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
            im = cv2.resize(image, (WIDTH, HEIGHT))
            cv2.imshow("Face Detection", im)
            labeled.append(im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        output = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (WIDTH, HEIGHT))
        for image in labeled:
            output.write(image)
        output.release()




if __name__ == '__main__':
    main()