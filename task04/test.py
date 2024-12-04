from inferemote.testing import AiremoteTest
from flowerRecog import FlowerRecog


class MyTest(AiremoteTest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """ An airemote object """
        self.air = FlowerRecog()

    """ Define a callback function for inferencing, which will be called for every single image """

    def run(self, image):
        n = self.air.inference_remote(image)
        shape = image.shape[:2]
        new_image = self.air.make_image(n, shape)
        return new_image


if __name__ == "__main__":
    """ default image for testing """

    t = MyTest(remote="localhost")
    t.start(input="/Users/qi7876/Desktop/roses_97.jpg", mode="show", wait=5)
