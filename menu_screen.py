import cv2
import numpy as np

class MenuScreen:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.selected_option = 0

    def run(self):
        while True:
            frame = self.build_frame()
            cv2.imshow('HoloDraw Launcher', frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('1'):
                return "HoloDraw"
            elif key == ord('2'):
                return "HoloPaint"
            
            if key == ord('q'):
                return None
            
    def build_frame(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (18,18,22)

        # Title
        cv2.putText(frame, 'HoloDraw', (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Select a mode to continue:', (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)

        # Option Cards
        self._draw_card(frame, 80, 240, 420, 220, "1", "HoloDraw", "Freehand sketching with gesture tools")
        self._draw_card(frame, 540, 240, 420, 220, "2", "HoloPaint", "Upload an image and paint inside the outlines")

        cv2.putText(frame, "Press 1 or 2, or use gestures later inside the app", (80,self.height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)

        return frame
