import cv2

img = cv2.imread('output/image0.png')
if img is None:
    raise Exception("画像が読み込めませんでした。ファイルパスを確認してください。")
height, width, layers = img.shape
size = (width, height)


name = 'project.mp4'
out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'MP4V'), 5.0, size)

for i in range(400):
    filename = f"output/image{i}.png"
    img = cv2.imread(filename)
    
    if img is None:
        continue
    
    out.write(img)

out.release()
