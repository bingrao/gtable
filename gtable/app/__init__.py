from gtable.app.ctgan import CTGANSynthesizer
from gtable.app.tvae import TVAESynthesizer
from gtable.app.gtable.gtable import GTABLESynthesizer
from gtable.app.tablegan import TableganSynthesizer

str2app = {"CTGAN": CTGANSynthesizer,
           "TVAE": TVAESynthesizer,
           "GTABLE": GTABLESynthesizer,
           "TABLEGAN": TableganSynthesizer}

__all__ = ['str2app',
           'CTGANSynthesizer',
           'TVAESynthesizer',
           'GTABLESynthesizer',
           'TableganSynthesizer']
