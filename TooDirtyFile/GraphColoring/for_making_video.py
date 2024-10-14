import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm
import time
    
def broken(i,k,result,h,w):
    i = i*w + k
    colors = len(result[0])
    cnt = 0
    for col in range(colors):
        if result[i][col] == 1:
            if i>=w:
                if result[i-w][col] == 1:
                    return "D"
            if i%w != 0:
                if result[i-1][col] == 1:
                    return "D"
            if i%w != w-1:
                if result[i+1][col] == 1:
                    return "D"
            if i<h*w-w:
                if result[i+w][col] == 1:
                    return "D"
            cnt += 1
    if cnt != 1:
        return "N"
    return "O"

def split_every_4element(li):
    splited = [li[i: i+4] for i in range(0, len(li), 4)]
    return splited

if __name__ == '__main__':

    csv_file = 'all_bit.csv'

    array = []

    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            array.append([int(x) for x in row])
    array_new = []
    for i in range(len(array)):
        array_new.append(split_every_4element(array[i]))

    h = 20
    w = 20
    c = 4

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].set_title("")
    ax[0].axis('off')
    ax[0].set_xticks(np.arange(w+2))
    ax[0].set_yticks(np.arange(h+2))
    ax[0].set_xticklabels(np.arange(2, w+4))
    ax[0].set_yticklabels(np.arange(2, h+4))
    plt.gca().invert_yaxis()

    ax[1].set_title("")
    ax[1].axis('off')
    ax[1].set_xticks(np.arange(w+2))
    ax[1].set_yticks(np.arange(h+2))
    ax[1].set_xticklabels(np.arange(2, w+4))
    ax[1].set_yticklabels(np.arange(2, h+4))    
    ax[1].text(0, -h*0.08,"  ", ha='center', va='center', fontsize=5, bbox=dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.5))
    ax[1].text(w*0.20, -h*0.08,"broken duplicate constraint", ha='center', va='center', fontsize=5)
    ax[1].text(0, -h*0.12,"  ", ha='center', va='center', fontsize=5, bbox=dict(facecolor="red", edgecolor='white', boxstyle='round', pad=0.5))
    ax[1].text(w*0.20, -h*0.12,"broken one-hot constraint", ha='center', va='center', fontsize=5)
    plt.gca().invert_yaxis()

    color = ["red", "blue", "green", "yellow"]
    an_step = 1000

    text_bits1 = [[[None for _ in range(c)] for _ in range(w)] for _ in range(h)]
    text_bits2 = [[[None for _ in range(c)] for _ in range(w)] for _ in range(h)]
    
    progress = ax[0].text(0, 0, "", fontsize=5, bbox=dict(facecolor='white', edgecolor='white', boxstyle='round', pad=0))


    for cnt in tqdm(range(an_step), desc="Processing", unit="iterations"):
        # mc = cnt % mc_step
        an = int(cnt)
        for i in range(h):
            for k in range(w):
                Is_broken = broken(i,k,array_new[cnt],h,w)
                for j in range(c):
                    if (Is_broken == "D"):
                        if text_bits2[i][k][j] is None: 
                            text_bits2[i][k][j] = ax[1].text(1+k, 1+i, "  ", ha='center', va='center', fontsize=5,bbox=dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits2[i][k][j].set_text("  ")
                            text_bits2[i][k][j].set_bbox(dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.5))    

                    elif (Is_broken == "N"):
                        if text_bits2[i][k][j] is None: 
                            text_bits2[i][k][j] = ax[1].text(1+k, 1+i, "  ", ha='center', va='center', fontsize=5,bbox=dict(facecolor="red", edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits2[i][k][j].set_text("  ")
                            text_bits2[i][k][j].set_bbox(dict(facecolor="red", edgecolor='white', boxstyle='round', pad=0.5))    

                    elif (Is_broken == "O"):
                        if text_bits2[i][k][j] is not None:
                            text_bits2[i][k][j].set_text("")
                            text_bits2[i][k][j].set_bbox(dict(facecolor="white", edgecolor='white', boxstyle='round', pad=0))
                    
                    if array_new[cnt][i*w+k][j] == 1:
                        if text_bits1[i][k][j] is None:
                            text_bits1[i][k][j] = ax[0].text(1+k, 1+i, j, ha='center', va='center', fontsize=5,bbox=dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits1[i][k][j].set_text(j)
                            text_bits1[i][k][j].set_bbox(dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=0.5))
                    else:
                        if text_bits1[i][k][j] is not None:
                            text_bits1[i][k][j].set_text("")
                            text_bits1[i][k][j].set_bbox(dict(facecolor="white", edgecolor='white', boxstyle='round', pad=0))


        progress.set_text(f"anneal step {an}/{an_step}")
# \n monte carlo step {mc}/{mc_step}
        plt.savefig("output/image" + str(cnt) + ".png", dpi=300)


    plt.close()
