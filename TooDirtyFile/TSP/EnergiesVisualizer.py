import matplotlib.pyplot as plt
import numpy as np
import csv

if __name__ == "__main__":

    csv_file = ['energies.csv','driver_energies.csv']
    titles = [['Anneal Steps VS Energies of Objective','anneal step','qubo energy'],['Anneal Steps VS Driver Hamiltonian','anneal step','driver hamiltonian energy']]
    
    for que in range(2):
        array = []

        with open(csv_file[que], newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                array.append([float(x) for x in row])

        plt.figure(figsize=(10, 6))

        for i, row in enumerate(array):
            plt.plot(row)

        plt.title(titles[que][0])
        plt.xlabel(titles[que][1])
        plt.ylabel(titles[que][2])
        plt.legend(loc='best')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    
