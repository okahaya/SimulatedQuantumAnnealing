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

    coords = []
    with open('sites.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            coords.append((float(row[0]), float(row[1])))

    visit_order = []
    with open('allroute.csv','r') as file:
        reader = csv.reader(file)
        for row in reader:
            visit_order.append([int(val) for val in row])
    route = [[] for i in range(10000)]
    for j in range(10000):
        for i in range(len(coords)):
            route[j].append(coords[visit_order[j][i]])

    for i in range(10000):
        route[i].append(route[i][0])

    energies = []
    with open('energies.csv', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                energies = [float(x) for x in row]
                break
                

    an_step = 10000
    # mc_step = 100
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    for idx, (x, y) in enumerate(coords):
        ax[0].text(x, y, f' {idx}', fontsize=12, color='black')
    
    ax[1].set_title('Anneal Steps VS Energies of Objective')
    ax[1].set_xlabel('monte carlo step')
    ax[1].set_ylabel('qubo energy')
    line = ax[1].plot([], [], color='red')[0] 

    ax[1].autoscale(enable=True, axis='both')
    ax[1].set_autoscaley_on(True) 

    x_data = []
    y_data = []

    for cnt in tqdm(range(an_step),desc="Processing", unit="iterations"):
        # mc = cnt % mc_step
        an = cnt
        
        total_distance = 0

        x_vals, y_vals = zip(*route[cnt])
        
        edge, = ax[0].plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
        e1 = ax[0].scatter(*zip(*coords), color='r')
        progress = ax[0].text(0,0,f"anneal step {an}/{an_step}",fontsize=10,bbox=dict(facecolor='white', edgecolor='white', boxstyle='round', pad=0))
        # \n monte carlo step {mc}/{mc_step}
        x_data.append(cnt)
        y_data.append(energies[cnt])
        line.set_data(x_data,y_data)
        ax[1].relim()  
        ax[1].autoscale_view() 
        # ax[1].plot(cnt,energies[cnt],color='red')
        plt.plot()

        
        plt.savefig("output/image"+str(cnt  )+".png", dpi=300)

        e1.remove()
        edge.remove()
        progress.remove()
        