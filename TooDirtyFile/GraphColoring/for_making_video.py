import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv
from tqdm import tqdm
import time

def split_every_4element(li):
    splited = [li[i: i+4] for i in range(0, len(li), 4)]
    return splited

if __name__ == '__main__':
    # CSVファイルのパス
    csv_file = 'all_bit.csv'

    # 空の二次元配列を作成
    array = []

    # CSVファイルを読み込む
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # 行ごとにリストに変換して二次元配列に追加
            array.append([int(x) for x in row])
    array_new = []
    for i in range(len(array)):
        array_new.append(split_every_4element(array[i]))

    h = int(len(array)**(1/2))
    w = h
    c = len(array[0])

    fig, ax = plt.subplots()
    ax.set_title("")
    ax.axis('off')
    ax.set_xticks(np.arange(4+4))
    ax.set_yticks(np.arange(4+2))
    ax.set_xticklabels(np.arange(1, 4+5))
    ax.set_yticklabels(np.arange(2, 4+4))
    plt.gca().invert_yaxis()
    color = ["red","blue","green","yellow","pink"]
    an_step = 20
    mc_step = 20

    for cnt in tqdm(range(mc_step*an_step),desc="Processing", unit="iterations"):
        mc = cnt % mc_step
        an = int(cnt/mc_step)
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    if array_new[cnt][i*4+k][j] == 1:
                        bits = ax.text(2+k, 1+i,j, ha='center', va='center', fontsize=20, bbox=dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=1))
        progress = ax.text(0,0,f"anneal step {an}/{an_step}\n monte carlo step {mc}/{mc_step}",fontsize=20,bbox=dict(facecolor='white', edgecolor='white', boxstyle='round', pad=0))

        #Gifアニメーションのために画像をためます
        plt.savefig("output/image"+str(cnt  )+".png", dpi=300)
        bits.remove()
        progress.remove()
    