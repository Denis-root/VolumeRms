from PIL import Image

# Abrir imagen original (PNG, JPG, etc.)
img = Image.open("6540842.png")

# Redimensionar a 256x256 píxeles (tamaño ideal para icono de Windows)
img = img.resize((256, 256), Image.LANCZOS)

# Guardar como .ico
img.save("logo.ico", format='ICO')
