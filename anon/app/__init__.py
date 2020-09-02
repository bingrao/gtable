from anon.app.imagegan import ImageGAN
from anon.app.tablegan import TableGAN

str2app = {"image": ImageGAN, "table": TableGAN}
__all__ = ["str2app", "ImageGAN", "TableGAN"]
