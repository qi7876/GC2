import numpy as np
import cv2 as cv

from inferemote.atlas_remote import AtlasRemote


class FlowerRecog(AtlasRemote):
    MODEL_WIDTH = 100
    MODEL_HEIGHT = 100

    def __init__(self, **kwargs):
        super().__init__(port=7777, **kwargs)

    def pre_process(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.MODEL_WIDTH, self.MODEL_HEIGHT)).copy()
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return image.tobytes()

    def post_process(self, result):
        """Decode to image after net travelling"""
        blob = np.frombuffer(result[0], np.float32)

        vals = blob.flatten()
        max_val = np.max(vals)
        vals = np.exp(vals - max_val)
        sum_val = np.sum(vals)
        vals /= sum_val
        top_k = vals.argsort()[-1:-6:-1]

        class_names = {
            0: "daisy",
            1: "dandelion",
            2: "roses",
            3: "sunflowers",
            4: "tulips",
        }

        predictedFlower = class_names.get(top_k[0])

        return predictedFlower

    def make_image(self, n, orig_shape):
        text = str(n)
        print("Flower:", text)
        image = np.zeros((orig_shape[0], orig_shape[1], 3), dtype=np.uint8)
        image = cv.putText(
            image,
            text,
            (10, 50),
            cv.FONT_HERSHEY_COMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        return image
