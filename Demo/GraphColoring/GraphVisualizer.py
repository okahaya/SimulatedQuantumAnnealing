import matplotlib.pyplot as plt
import numpy as np
import csv

def visualize_colored_graph(result, size_w,size_h, colors):
    fig, ax = plt.subplots()
    ax.set_title(f"Colored Graph")
    ax.axis('off')
    ax.set_xticks(np.arange(size_w+2))
    ax.set_yticks(np.arange(size_h+2))
    ax.set_xticklabels(np.arange(2, size_w+4))
    ax.set_yticklabels(np.arange(2, size_h+4))
    color = ["red","blue","green","yellow","pink"]
    for i in range(size_h):
        for k in range(size_w):
            for j in range(colors):
                if result[i*size_w+k][j] == 1:
                    ax.text(1+k, 1+i,j, ha='center', va='center', fontsize=5, bbox=dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=0.5))

 
    plt.gca().invert_yaxis()
    plt.show()

def broken(i,k,result,h,w):
    i = i*w + k
    colors = len(result[0])
    for col in range(colors):
        if result[i][col] == 1:
            if i>=w:
                if result[i-w][col] == 1:
                    return True
            if i%w != 0:
                if result[i-1][col] == 1:
                    return True
            if i%w != w-1:
                if result[i+1][col] == 1:
                    return True
            if i<h*w-w:
                if result[i+w][col] == 1:
                    return True
    return False

def only_broken_visualize_colored_graph(result, size_w,size_h, colors):
    fig, ax = plt.subplots()
    ax.set_title(f"Only Broken")
    ax.axis('off')
    ax.set_xticks(np.arange(size_w+2))
    ax.set_yticks(np.arange(size_h+2))
    ax.set_xticklabels(np.arange(2, size_w+4))
    ax.set_yticklabels(np.arange(2, size_h+4))
    for i in range(size_h):
        for k in range(size_w):
            for j in range(colors):
                if result[i*size_w+k][j] == 1:
                    if (broken(i,k,result,size_h,size_w) == True):
                        ax.text(1+k, 1+i,j, ha='center', va='center', fontsize=5, bbox=dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.5))

 
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':
    # CSVファイルのパス
    csv_file = 'graphcolored.csv'

    # 空の二次元配列を作成
    array = []

    # CSVファイルを読み込む
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # 行ごとにリストに変換して二次元配列に追加
            array.append([int(x) for x in row])

    h = int(len(array)**(1/2))
    w = h
    c = len(array[0])
    
    visualize_colored_graph(array,h,w,c)
    only_broken_visualize_colored_graph(array,h,w,c)