import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm
import time

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

    fig, ax = plt.subplots()
    ax.set_title("")
    ax.axis('off')
    ax.set_xticks(np.arange(w+4))
    ax.set_yticks(np.arange(h+2))
    ax.set_xticklabels(np.arange(1, w+5))
    ax.set_yticklabels(np.arange(2, h+4))
    plt.gca().invert_yaxis()
    color = ["red", "blue", "green", "yellow"]
    an_step = 100
    mc_step = 100

    text_bits = [[[
        ax.text(2+k, 1+i, "", ha='center', va='center', fontsize=5,
                bbox=dict(facecolor='white', edgecolor='white', boxstyle='round', pad=1))
        for j in range(c)] for k in range(w)] for i in range(h)]
    
    progress = ax.text(0, 0, "", fontsize=20, bbox=dict(facecolor='white', edgecolor='white', boxstyle='round', pad=0))

    for cnt in tqdm(range(mc_step * an_step), desc="Processing", unit="iterations"):
        mc = cnt % mc_step
        an = int(cnt / mc_step)
        for i in range(h):
            for k in range(w):
                for j in range(c):
                    if array_new[cnt][i*w+k][j] == 1:
                        text_bits[i][k][j].set_text(j)
                        text_bits[i][k][j].set_bbox(dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=1))
                    else:
                        text_bits[i][k][j].set_text("") 

        progress.set_text(f"anneal step {an}/{an_step}\n monte carlo step {mc}/{mc_step}")

        plt.savefig("output/image" + str(cnt) + ".png", dpi=300)

    plt.close()
