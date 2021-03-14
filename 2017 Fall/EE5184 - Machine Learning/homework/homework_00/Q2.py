import sys

from PIL import Image
inputFile = str(sys.argv[1])
image = Image.open(inputFile)
p = image.load()
r, g, b = image.split()
rp = r.load()
gp = g.load()
bp = b.load()

for i in range(image.size[0]):   
    for j in range(image.size[1]):   
        p[i, j] = (int(rp[i, j] / 2), int(gp[i, j] / 2), int(bp[i, j] / 2)) 
        image.putpixel((i, j), p[i, j])
image.save('Q2.png')
