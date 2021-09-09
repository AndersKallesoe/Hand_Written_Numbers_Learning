import numpy as np
import cv2
from network import Network

class Draw:
    def __init__(self, network):
        self.network = network
    blank = np.zeros((56, 56), np.uint8)
    img = blank.copy()
    current_img = blank.copy()
    draw = False
    ix = -1
    iy = -1

    def draw_circle(self, event, x, y, param, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
            self.ix, iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                self.img = cv2.circle(self.img, (x, y), 1, 255, -1)
                overlays_base = self.current_img.copy()
                for i in range(5): # number of iterations and opac value that gave the desired result during testing
                    overlay = overlays_base.copy()
                    opac = 0.7
                    cv2.circle(overlay, (x, y), 1, 255, 2)
                    self.current_img = cv2.addWeighted(self.current_img, opac, overlay, 1 - opac, 0)
                    self.img = self.current_img
        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.rectangle(self.img, (512, 512), (0, 0), 0, -10)

    def start(self):
        cv2.namedWindow("Numberdrawing", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Numberdrawing", (512, 512))
        cv2.setMouseCallback('Numberdrawing', self.draw_circle)

        printed = set([])
        while True:
            try:
                cv2.getWindowProperty('Numberdrawing', 0)
            except cv2.Error:
                break
            cv2.imshow('Numberdrawing', self.img)
            res = cv2.waitKey(1)
            if res & 0xFF == 27:
                break
            if res & 0xFF == 32:
                input = cv2.resize(self.img, (28, 28), interpolation=cv2.INTER_LINEAR).flatten()
                output, _, _ = self.network.compute_output(input)
                output = np.argmax(output)
                print(output)

        cv2.destroyAllWindows()
