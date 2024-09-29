import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm
import time
import csv

def split_every_Nelement(li,N):
    splited = [li[i: i+N] for i in range(0, len(li), N)]
    return splited

if __name__ == '__main__':
    
    csv_file = 'bit.csv'
    array = []

    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            array.append([int(x) for x in row])
    array_new = []
    n = int((len(array[0]))**(1/2))
    for i in range(len(array)):
        array_new.append(split_every_Nelement(array[i],n))

    coords = []
    with open('sites.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            coords.append((float(row[0]), float(row[1])))


    an_step = 100
    mc_step = 100
    ims = []
    fig, ax = plt.subplots()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("")
    for idx, (x, y) in enumerate(coords):
        plt.text(x, y, f' {idx}', fontsize=12, color='black')
    
    for cnt in tqdm(range(mc_step*an_step),desc="Processing", unit="iterations"):
        mc = cnt % mc_step
        an = int(cnt/mc_step)
        route = []
        for j in range(len(coords)):
            for i in range(len(coords)):
                if array_new[cnt][i][j] == 1:
                    route.append(coords[i])
        route.append(route[0])        
        total_distance = 0

        x_vals, y_vals = zip(*route)
        
        edge, = plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
        e1 = plt.scatter(*zip(*coords), color='r')


        plt.grid(True)
        plt.plot()

        progress = ax.text(0,0,f"anneal step {an}/{an_step}\n monte carlo step {mc}/{mc_step}",fontsize=20,bbox=dict(facecolor='white', edgecolor='white', boxstyle='round', pad=0))
        ims.append(plt.plot())
        
        plt.savefig("output/image"+str(cnt  )+".png", dpi=300)

        e1.remove()
        edge.remove()
        progress.remove()
        