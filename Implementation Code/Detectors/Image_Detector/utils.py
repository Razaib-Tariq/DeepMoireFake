from PIL import Image
from PIL import ImageOps
import torchvision.transforms.functional as F
import torch

def isotropically_resize_image_pil(img, size, 
                                   interpolation_down=Image.BICUBIC, 
                                   interpolation_up=Image.BILINEAR):
    w, h = img.size

    if max(w, h) == size:
        return img

    if w > h:
        scale = size / w
        h = int(h * scale)
        w = size
    else:
        scale = size / h
        w = int(w * scale)
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down

    return img.resize((w, h), interpolation)

class IsotropicResizeTorch:
    def __init__(self, max_side, interpolation_down=Image.BICUBIC, interpolation_up=Image.BILINEAR):
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def __call__(self, img):
        return isotropically_resize_image_pil(img, size=self.max_side, 
                                              interpolation_down=self.interpolation_down, 
                                              interpolation_up=self.interpolation_up)

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_side={0})"
        return format_string.format(self.max_side)


class PadIfNeeded:
    def __init__(self, min_height, min_width, fill=0, padding_mode='constant'):
        self.min_height = min_height
        self.min_width = min_width
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # Calculate padding
        if img.size[0] < self.min_width or img.size[1] < self.min_height:
            padding_ltrb = self.calculate_padding(img)
            if self.padding_mode == 'constant':
                img = ImageOps.expand(img, border=padding_ltrb, fill=self.fill)
            else:
                img = F.pad(img, padding_ltrb, fill=self.fill, padding_mode=self.padding_mode)
        return img

    def calculate_padding(self, img):
        left = (self.min_width - img.size[0]) // 2
        right = self.min_width - img.size[0] - left
        top = (self.min_height - img.size[1]) // 2
        bottom = self.min_height - img.size[1] - top
        return (left, top, right, bottom)

    def __repr__(self):
        return self.__class__.__name__ + f"(min_height={self.min_height}, min_width={self.min_width}, fill={self.fill}, padding_mode={self.padding_mode})"
class ToIntImage:
    def __call__(self, tensor):
        # Clamp to ensure values stay in the [0, 255] range
        # This might be especially useful if there were any other transformations before
        tensor = torch.clamp(tensor * 255, 0, 255)
        
        # Convert to numpy and then to uint8 type
        return tensor.float()


if __name__ == "__main__":
    # Load image
    image_path = "/00000.png"
    img = Image.open(image_path)
    print("Original image: ", img.size)
    # Define transformation
    transform = IsotropicResizeTorch(256)

    # Apply transformation
    resized_img = transform(img)
    print("Tranformed image after IsotropicResizeTorch:", resized_img.size)

    print("Original image: ", img.size)
    # Define transformation
    transform = IsotropicResizeTorch(256)
    # Apply transformation
    resized_img = transform(img)
    print("Tranformed image after IsotropicResizeTorch:", resized_img.size)
    # Test the PadIfNeeded transform
    transform = PadIfNeeded(256, 256, fill=(0, 0, 0))  # Black padding
    paded_img = transform(resized_img)
    print("Tranformed image after PadIfNeeded:", paded_img.size)
    paded_img.save('dummy.jpg', 'JPEG') 

