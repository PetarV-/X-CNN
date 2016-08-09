import colorsys
import numpy as np

def get_palette(size):
    step=1/float(size)
    palette=np.zeros((size,3))
    hue=0
    counter=0
    while hue<1:
        palette[counter,:]=colorsys.hsv_to_rgb(hue,1,1)
	#print("hue:",hue,"color",colorsys.hsv_to_rgb(hue,1,1))
        counter+=1
        hue+=step
    return palette

#palette=get_palette(8)
#print(palette)
