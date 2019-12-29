import numpy as np
try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }


class frame_queue(object):
    """
    push current img to a queue
    img: img input for the moment
    """
    def __init__(self, nb_frames, target_size):
        self.batch = np.zeros((1, nb_frames) + target_size + (3,))
        self.target_size = target_size
    def img_resiz(self,img, interpolation = 'nearest'):

        img = pil_image.fromarray(img) # change image format from nparray to jpg
        if self.target_size is not None:
            width_height_tuple = (self.target_size[1], self.target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
        return img


    def img_to_array(self,img):
        img_array = np.asarray(img, dtype = 'float32')
        return img_array

    def img_inQueue(self, img):
        for i in range(self.batch.shape[1] - 1):
            self.batch[0, i] = self.batch[0, i+1]
        img = self.img_resiz(img)
        img = self.img_to_array(img)
        x = self.img_to_array(img)
        self.batch[0, self.batch.shape[1] - 1] = x / 255

        return self.batch
