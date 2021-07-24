import numpy as np
import cv2
from network import Network
network = Network()

blank = np.zeros((56, 56), np.uint8)
img = blank.copy()
current_img = blank.copy()
draw = False
ix = -1
iy = -1

def euc_dist(center, p):
    return np.sqrt((center[0] - p[0])**2) + np.sqrt((center[0] - p[0])**2)


def draw_circle(event, x, y, param, flags):
    global ix, iy, draw, img, current_img
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        ix, iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        center = (x, y)

        if draw:
            img = cv2.circle(img, (x, y), 1, 255, -1)
            overlays_base = current_img.copy()
            for i in range(5):
                overlay = overlays_base.copy()
                opac = 0.7
                cv2.circle(overlay, (x, y), 1, 255, 2)
                current_img = cv2.addWeighted(current_img, opac, overlay, 1 - opac, 0)
                img = current_img
            # overlays_base = current_img.copy()
            # for i in range(1, 16):
            #     overlay = overlays_base.copy()
            #     opac = 0.7
            #     cv2.circle(overlay, (x, y), i, 255, -1)
            #     current_img = cv2.addWeighted(current_img, opac, overlay, 1 - opac, 0)
            #     img = current_img
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.rectangle(img, (512, 512), (0, 0), 0, -10)


cv2.namedWindow("Numberdrawing", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Numberdrawing", (512, 512))
cv2.setMouseCallback('Numberdrawing', draw_circle)

printed = set([])
while True:
    cv2.imshow('Numberdrawing', img)
    res = cv2.waitKey(1)
    if res & 0xFF == 27:
        break
    if res & 0xFF == 32:
        input = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR).flatten()
        output, _, _ = network.compute_output(input)
        output = np.argmax(output)
        print(output)

cv2.destroyAllWindows()
