from anon.app.imageapp import ImageGAN
from anon.app.tableapp import TableGAN

str2app = {"image": ImageGAN, "table": TableGAN}
__all__ = ["str2app", "ImageGAN", "TableGAN"]
