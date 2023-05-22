import numpy as np
import numba
import cv2

from dsec_det.label import COLORS, CLASSES


@numba.jit(nopython=True)
def render_events_on_image(image, x, y, p):
    for x_, y_, p_ in zip(x, y, p):
        if p_ == 0:
            image[y_, x_] = np.array([0, 0, 255])
        else:
            image[y_, x_] = np.array([255, 0, 0])
    return image

def render_object_detections_on_image(img, tracks, **kwargs):
    return _draw_bbox_on_img(img, tracks['x'], tracks['y'], tracks['w'], tracks['h'],
                             tracks['class_id'], **kwargs)

def _draw_bbox_on_img(img, x, y, w, h, labels, scores=None, conf=0.5, label="", scale=1, linewidth=2, show_conf=True):
    for i in range(len(x)):
        if scores is not None and scores[i] < conf:
            continue

        x0 = int(scale * (x[i]))
        y0 = int(scale * (y[i]))
        x1 = int(scale * (x[i] + w[i]))
        y1 = int(scale * (y[i] + h[i]))
        cls_id = int(labels[i])

        color = (COLORS[cls_id] * 255).astype(np.uint8).tolist()

        #track_id = box['track_id']
        text = f"{label}-{CLASSES[cls_id]}"

        if scores is not None and show_conf:
            text += f":{scores[i] * 100: .1f}"

        txt_color = (0, 0, 0) if np.mean(COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, linewidth)

        txt_bk_color = (COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        txt_height = int(1.5*txt_size[1])
        cv2.rectangle(
            img,
            (x0, y0 - txt_height),
            (x0 + txt_size[0] + 1, y0 + 1),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]-txt_height), font, 0.4, txt_color, thickness=1)
    return img
