import cv2
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    detector = mp.solutions.face_detection
    drawer = mp.solutions.drawing_utils

    with detector.FaceDetection(model_selection=0, min_detection_confidence=.5) as face:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to capture image")
                continue
            # annotated_image = image.copy()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.detections:
                for detection in results.detections:
                    drawer.draw_detection(image, detection)
            im = cv2.resize(image, (1600, 1200))
            cv2.imshow("Face Detection", im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()