import cv2
import glob

img = cv2.imread('output/image0.png')
height, width, layers = img.shape
size = (width, height)
img_array = []
for i in range(100):
    filename = f"output/image{i}.png"
    img = cv2.imread(filename)
    img_array.append(img)

name = 'project.mp4'
out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'MP4V'), 5.0, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()