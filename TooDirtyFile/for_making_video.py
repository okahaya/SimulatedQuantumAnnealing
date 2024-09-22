import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv


def visualize_colored_graph(result, size_w,size_h, colors):
    fig, ax = plt.subplots()
    ax.set_title("")
    ax.axis('off')
    ax.set_xticks(np.arange(size_w+4))
    ax.set_yticks(np.arange(size_h+2))
    ax.set_xticklabels(np.arange(1, size_w+5))
    ax.set_yticklabels(np.arange(2, size_h+4))
    color = ["red","blue","green","yellow","pink"]
    for i in range(size_h):
        for k in range(size_w):
            for j in range(colors):
                if result[i*size_w+k][j] == 1:
                    ax.text(2+k, 1+i,j, ha='center', va='center', fontsize=20, bbox=dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=0))

 
    plt.gca().invert_yaxis()
    plt.plot()


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

    ims = []
    for cnt in range(int(len(array)/4)):
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    if array_new[cnt][i*4+k][j] == 1:
                        ax.text(2+k, 1+i,j, ha='center', va='center', fontsize=20, bbox=dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=1))
        ims.append(plt.plot())

        fig1=plt.pause(0.001)
        #Gifアニメーションのために画像をためます
        plt.savefig("output/image"+str(cnt  )+".png", dpi=300)
        # dst = cv2.imread('output/image'+str(a)+'.png')
        # out.write(dst) #mp4やaviに出力します


    # 見本
    # for i in range(10):
    #     rand = np.random.randn(100)     # 100個の乱数を生成
    #     im = plt.plot(rand)             # 乱数をグラフにする
    #     ims.append(im)                  # グラフを配列 ims に追加
    # 10枚のプロットを 100ms ごとに表示
    # ani = animation.ArtistAnimation(fig, ims, interval=100)
    # plt.show()
    
    # visualize_colored_graph(array,h,w,c)