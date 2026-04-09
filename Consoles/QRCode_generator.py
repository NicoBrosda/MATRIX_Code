import qrcode
import qrcode.image.svg as svg
from pathlib import Path

# Link of webpage
data1 = 'https://ruhr-uni-bochum.sciebo.de/s/QCNzj7doDCW2Zwe'

# Setting up QR code type and factory
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_Q,
    box_size=10,
    border=4,
)
qr.add_data(data1)
qr.make(fit=True)
factory = svg.SvgPathImage

# Create image of QR code
img = qr.make_image(fill_color="#003560", back_color="white", image_factory=factory)  # Colour can only be set without .svg!

# Save image under given path
img.save(Path("/Users/nico_brosda/Desktop/Conferences/DPG 2026 Dresden/qrcode_s_timescale.svg"))

# Second link
data2 = 'https://ruhr-uni-bochum.sciebo.de/s/eJSQDeXpKjY9WMN'

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_Q,
    box_size=10,
    border=4,
)
qr.add_data(data2)
qr.make(fit=True)
factory = svg.SvgPathImage

# Second QR code a image + saved
img = qr.make_image(fill_color="#003560", back_color="white", image_factory=factory)
img.save(Path("/Users/nico_brosda/Desktop/Conferences/DPG 2026 Dresden/qrcode_ms_timescale.svg"))
