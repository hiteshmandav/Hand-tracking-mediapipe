import cv2 as cv
import mediapipe as mp
import time

class HandDetector():
    
    def __init__(self,
                 static_mode=False, 
                 max_hands = 2, 
                 min_detection_confidence = 0.5, 
                 min_tracking_confidence = 0.5):
        
        # self.static_mode = static_mode
        # self.max_hands = max_hands
        # self.min_detection_confidence = min_detection_confidence
        # self.min_tracking_confidence = min_tracking_confidence

        #initalizing mediapipe
        self.mp_hands = mp.solutions.hands
        # self.hand = self.mp_hands.Hands(self.static_mode, 
        #                                 self.max_hands, 
        #                                 self.min_detection_confidence, 
        #                                 self.min_tracking_confidence)
        self.hand = self.mp_hands.Hands(static_mode, 
                                        max_hands, 
                                        min_detection_confidence, 
                                        min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        
    def findHands(self, frame, draw=True):
        frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.result = self.hand.process(frame_RGB)
    
        if self.result.multi_hand_landmarks:
            # itrating on each hand 
            for handLandmark in self.result.multi_hand_landmarks:
                # drawing connections on each hand 
                self.mp_draw.draw_landmarks(frame, handLandmark , self.mp_hands.HAND_CONNECTIONS)
        # return
    
    def findFingerPosition(self, frame):
        # itrating on each point on hand
        # for id, landmark in enumerate(handLandmark.landmark):
        #     height,  width, channel = frame.shape
        #     cx, cy = int(landmark.x * width), int(landmark.y * height)
        #     if id == 8:
        #         cv.circle(frame, (cx, cy), 10, (255, 0, 0), 5)
        return

def main():
    #input variables
    camera_number = 1
    frame_width = 640
    frame_height = 480
    p_time = 0
    c_time = 0


    #video cap
    capture = cv.VideoCapture(camera_number)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    while True:
        _, frame = capture.read()
        hd = HandDetector();
        hd.findHands(
            frame=frame
        )
        
        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time
        
        cv.putText(frame, f'FPS :: {int(fps)}', (10, 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)    
        cv.imshow('TRACKER' , frame)
        key = cv.waitKey(1)
        if key == 27:
            break;
        
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()