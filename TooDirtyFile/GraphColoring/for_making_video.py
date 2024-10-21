import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm
import time
import matplotlib.gridspec as gridspec
    
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

    csv_file1 = 'bit_swaq.csv'
    csv_file2 = 'bit_penalty.csv'
    
    array1 = []
    array2 = []

    with open(csv_file1, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            array1.append([int(x) for x in row])
    array_new1 = []
    for i in range(len(array1)):
        array_new1.append(split_every_4element(array1[i]))

    with open(csv_file2, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            array2.append([int(x) for x in row])
    array_new2 = []
    for i in range(len(array2)):
        array_new2.append(split_every_4element(array2[i]))

    energies1 = []
    with open('energies_swaq.csv', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                energies1 = [float(x) for x in row]
                break
    energies2 = []
    with open('energies_penalty.csv', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                energies2 = [float(x) for x in row]
                break
    h = 20
    w = 20
    c = 4

    fig = plt.figure(constrained_layout=True,figsize=(10,5))
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])

    # fig, ax = plt.subplots(2, 4, gridspec_kw={'width_ratios': [1, 1, 1, 1], 'height_ratios': [1, 1]},figsize=(10,5))
    ax = {
            (0,0):fig.add_subplot(gs[0, 0:2]),
            (0,1):fig.add_subplot(gs[0, 2:4]),
            (1,0):fig.add_subplot(gs[1, 0]),
            (1,1):fig.add_subplot(gs[1, 1]),
            (1,2):fig.add_subplot(gs[1, 2]),
            (1,3):fig.add_subplot(gs[1, 3])
          }
    fig.tight_layout(pad=1.0)

    # ax[0, 1].axis('off') 
    line1 = ax[0,0].plot([], [], color='red')[0] 
    line2 = ax[0,1].plot([], [], color='red')[0] 

    ax[0,0].autoscale(enable=True, axis='both')
    ax[0,0].set_autoscaley_on(True) 
    ax[0,1].autoscale(enable=True, axis='both')
    ax[0,1].set_autoscaley_on(True) 

    x_data1 = []
    y_data1 = []
    x_data2 = []
    y_data2 = []

    ax[1,0].set_title("")
    ax[1,0].axis('off')
    ax[1,0].set_xticks(np.arange(w))
    ax[1,0].set_yticks(np.arange(h))
    ax[1,0].set_xticklabels(np.arange(0, w))
    ax[1,0].set_yticklabels(np.arange(0, h))
    plt.gca().invert_yaxis()

    ax[1,1].set_title("")
    ax[1,1].axis('off')
    ax[1,1].set_xticks(np.arange(w))
    ax[1,1].set_yticks(np.arange(h))
    ax[1,1].set_xticklabels(np.arange(0, w))
    ax[1,1].set_yticklabels(np.arange(0, h))    
    ax[1,1].text(0, -h*0.08,"  ", ha='center', va='center', fontsize=5, bbox=dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.1))
    ax[1,1].text(w*0.50, -h*0.08,"broken duplicate constraint", ha='center', va='center', fontsize=5)
    ax[1,1].text(0, -h*0.12,"  ", ha='center', va='center', fontsize=5, bbox=dict(facecolor="red", edgecolor='white', boxstyle='round', pad=0.1))
    ax[1,1].text(w*0.50, -h*0.12,"broken one-hot constraint", ha='center', va='center', fontsize=5)
    plt.gca().invert_yaxis()


    # ax[0, 3].axis('off') 

    ax[1,2].set_title("")
    ax[1,2].axis('off')
    ax[1,2].set_xticks(np.arange(w))
    ax[1,2].set_yticks(np.arange(h))
    ax[1,2].set_xticklabels(np.arange(0, w))
    ax[1,2].set_yticklabels(np.arange(0, h))
    plt.gca().invert_yaxis()

    ax[1,3].set_title("")
    ax[1,3].axis('off')
    ax[1,3].set_xticks(np.arange(w))
    ax[1,3].set_yticks(np.arange(h))
    ax[1,3].set_xticklabels(np.arange(0, w))
    ax[1,3].set_yticklabels(np.arange(0, h))    
    ax[1,3].text(0, -h*0.08,"  ", ha='center', va='center', fontsize=5, bbox=dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.1))
    ax[1,3].text(w*0.5, -h*0.08,"broken duplicate constraint", ha='center', va='center', fontsize=5)
    ax[1,3].text(0, -h*0.12,"  ", ha='center', va='center', fontsize=5, bbox=dict(facecolor="red", edgecolor='white', boxstyle='round', pad=0.1))
    ax[1,3].text(w*0.5, -h*0.12,"broken one-hot constraint", ha='center', va='center', fontsize=5)
    plt.gca().invert_yaxis()

    color = ["red", "blue", "green", "yellow"]
    an_step = 1000

    text_bits1 = [[[None for _ in range(c)] for _ in range(w)] for _ in range(h)]
    text_bits2 = [[[None for _ in range(c)] for _ in range(w)] for _ in range(h)]
    text_bits3 = [[[None for _ in range(c)] for _ in range(w)] for _ in range(h)]
    text_bits4 = [[[None for _ in range(c)] for _ in range(w)] for _ in range(h)]
    

    for cnt in tqdm(range(an_step), desc="Processing", unit="iterations"):
        # mc = cnt % mc_step
        x_data1.append(cnt)
        y_data1.append(energies1[cnt])
        x_data2.append(cnt)
        y_data2.append(energies2[cnt]+20**2*2)
        
        line1.set_data(x_data1,y_data1)
        ax[0,0].relim()  
        ax[0,0].autoscale_view() 
        line2.set_data(x_data2,y_data2)
        ax[0,1].relim()  
        ax[0,1].autoscale_view() 
        an = int(cnt)
        for i in range(h):
            for k in range(w):
                Is_broken = broken(i,k,array_new1[cnt],h,w)
                for j in range(c):
                    if (Is_broken == "N"):
                        if text_bits2[i][k][j] is None: 
                            text_bits2[i][k][j] = ax[1,1].text(1+k, 1+i, "  ", ha='center', va='center', fontsize=3,bbox=dict(facecolor="red", edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits2[i][k][j].set_text("  ")
                            text_bits2[i][k][j].set_bbox(dict(facecolor="red", edgecolor='white', boxstyle='round', pad=0.5))    

                    elif (Is_broken == "D"):
                        if text_bits2[i][k][j] is None: 
                            text_bits2[i][k][j] = ax[1,1].text(1+k, 1+i, "  ", ha='center', va='center', fontsize=3,bbox=dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits2[i][k][j].set_text("  ")
                            text_bits2[i][k][j].set_bbox(dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.5))    

                    
                    elif (Is_broken == "O"):
                        if text_bits2[i][k][j] is not None:
                            text_bits2[i][k][j].set_text("")
                            text_bits2[i][k][j].set_bbox(dict(facecolor="white", edgecolor='white', boxstyle='round', pad=0))
                    
                    if array_new1[cnt][i*w+k][j] == 1:
                        if text_bits1[i][k][j] is None:
                            text_bits1[i][k][j] = ax[1,0].text(1+k, 1+i, j, ha='center', va='center', fontsize=3,bbox=dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits1[i][k][j].set_text(j)
                            text_bits1[i][k][j].set_bbox(dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=0.5))
                    else:
                        if text_bits1[i][k][j] is not None:
                            text_bits1[i][k][j].set_text("")
                            text_bits1[i][k][j].set_bbox(dict(facecolor="white", edgecolor='white', boxstyle='round', pad=0))

                Is_broken = broken(i,k,array_new2[cnt],h,w)
                for j in range(c):
                    if (Is_broken == "N"):
                        if text_bits4[i][k][j] is None: 
                            text_bits4[i][k][j] = ax[1,3].text(1+k, 1+i, "  ", ha='center', va='center', fontsize=3,bbox=dict(facecolor="red", edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits4[i][k][j].set_text("  ")
                            text_bits4[i][k][j].set_bbox(dict(facecolor="red", edgecolor='white', boxstyle='round', pad=0.5))    

                    elif (Is_broken == "D"):
                        if text_bits4[i][k][j] is None: 
                            text_bits4[i][k][j] = ax[1,3].text(1+k, 1+i, "  ", ha='center', va='center', fontsize=3,bbox=dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits4[i][k][j].set_text("  ")
                            text_bits4[i][k][j].set_bbox(dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.5))    

                    elif (Is_broken == "O"):
                        if text_bits4[i][k][j] is not None:
                            text_bits4[i][k][j].set_text("")
                            text_bits4[i][k][j].set_bbox(dict(facecolor="white", edgecolor='white', boxstyle='round', pad=0))
                    
                    if array_new2[cnt][i*w+k][j] == 1:
                        if text_bits3[i][k][j] is None:
                            text_bits3[i][k][j] = ax[1,2].text(1+k, 1+i, j, ha='center', va='center', fontsize=3,bbox=dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits3[i][k][j].set_text(j)
                            text_bits3[i][k][j].set_bbox(dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=0.5))
                    else:
                        if text_bits3[i][k][j] is not None:
                            text_bits3[i][k][j].set_text("")
                            text_bits3[i][k][j].set_bbox(dict(facecolor="white", edgecolor='white', boxstyle='round', pad=0))


        plt.savefig("output/image" + str(cnt) + ".png", dpi=300)


    plt.close()
