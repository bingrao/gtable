from gtable.app.ctgan import CTGANSynthesizer
from gtable.app.tvae import TVAESynthesizer
from gtable.app.gtable import GTABLESynthesizer

str2app = {"CTGAN": CTGANSynthesizer, "TVAE": TVAESynthesizer, "GTABLE": GTABLESynthesizer}

__all__ = ['CTGANSynthesizer', 'TVAESynthesizer', 'GTABLESynthesizer', 'str2app']
