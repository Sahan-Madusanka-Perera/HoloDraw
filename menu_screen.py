import cv2
import numpy as np

class MenuScreen:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def run(self):
        cv2.namedWindow('HoloDraw Launcher')
        while True:
            frame = self.build_frame()
            cv2.imshow('HoloDraw Launcher', frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('1'):
                cv2.destroyWindow('HoloDraw Launcher')
                return "HoloDraw"
            elif key == ord('2'):
                cv2.destroyWindow('HoloDraw Launcher')
                return "HoloPaint"
            
            if key == ord('q'):
                cv2.destroyWindow('HoloDraw Launcher')
                return None
            
    def build_frame(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Deep slate background
        frame[:] = (22, 18, 15)

        # Title
        cv2.putText(frame, 'HoloDraw', (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 230, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Select a mode to continue:', (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        # Option Cards
        # HoloDraw -> Neon Pink Accent
        self._draw_card(frame, 80, 240, 450, 220, "1", "HoloDraw", "Freehand sketching with gesture tools", (200, 50, 255))
        # HoloPaint -> Electric Cyan Accent
        self._draw_card(frame, 540, 240, 500, 220, "2", "HoloPaint", "Upload an image and paint inside outlines", (255, 230, 0))

        cv2.putText(frame, "Press 1 or 2 to select, Q to quit", (80, self.height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)

        return frame

    def _draw_card(self, frame, x, y, w, h, shortcut, title, subtitle, accent_color):
        # Premium rounded card background
        self._draw_rounded_rect(frame, (x, y), (x + w, y + h), (32, 28, 24), -1, 15)
        self._draw_rounded_rect(frame, (x, y), (x + w, y + h), (70, 60, 55), 2, 15)

        # Sleek vertical accent stripe
        cv2.line(frame, (x + 15, y + 30), (x + 15, y + h - 30), accent_color, 4, cv2.LINE_AA)

        # Text with vibrant colors
        cv2.putText(frame, f"[{shortcut}]", (x + 35, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, accent_color, 3, cv2.LINE_AA)
        cv2.putText(frame, title, (x + 110, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (x + 35, y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (170, 170, 170), 1, cv2.LINE_AA)

    def _draw_rounded_rect(self, img, pt1, pt2, color, thickness, r):
        """Helper to draw a rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        if thickness < 0:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
        else:
            cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)