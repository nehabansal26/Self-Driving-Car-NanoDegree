import pandas as pd
import cv2
import matplotlib.image as mpimg
import numpy as np

df = pd.read_csv("/opt/data/driving_log.csv")

image = mpimg.read(df.center.iloc[0])
flip_img = np.fliplr(img)

image.

