# Global imports etc.

import numpy as np
from FEN_to_64grid import FEN_to_seq
#For Saving images as files
from PIL import Image
#For file path
import os
from os import listdir
#For plotting
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#Function to zoom in on a single space
def zoom_to_position(image, position):
    #Assert that image is correct size
    assert image.shape == (400, 400, 3), f"Image shape is {image.shape}, expected (400, 400, 3)"
    assert 1 <= position & position <= 64, f"Position is {position}, expected (1-64)"

    #from position to x/y
    x = (position-1)%8
    y = int((position-1)/8)

    #Make sub array
    sub_image_array = image[y*50:y*50+50, x*50:x*50+50, :]

    return sub_image_array