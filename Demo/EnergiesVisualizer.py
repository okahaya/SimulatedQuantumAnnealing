import matplotlib.pyplot as plt
import numpy as np
import csv

if __name__ == "__main__":
    # CSVファイルのパス
    csv_file = 'energies.csv'

    # 空の二次元配列を作成
    array = []

    # CSVファイルを読み込む
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # 行ごとにリストに変換して二次元配列に追加
            array.append([int(x) for x in row])

    # グラフを作成
    plt.figure(figsize=(10, 6))

    # 各行のデータをプロット
    for i, row in enumerate(array):
        plt.plot(row, label=f'Row {i+1}')

    # グラフの設定
    plt.title('Line Graph of Data')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend(loc='best')
    plt.grid(True)

    # グラフの表示
    plt.tight_layout()
    plt.show()
