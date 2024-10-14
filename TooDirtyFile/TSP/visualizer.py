import matplotlib.pyplot as plt
import numpy as np
import csv

coords = []
with open('sites.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        coords.append((float(row[0]), float(row[1])))

# visit_order = []
# with open('TSP.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         visit_order.append([int(val) for val in row])

# visit_order = np.array(visit_order)
# route = []
# for j in range(len(coords)):
#     for i in range(len(coords)):
#         if visit_order[i][j] == 1:
#             route.append(coords[i])
visit_order = []
with open('allroute.csv','r') as file:
    reader = csv.reader(file)
    for row in reader:
        visit_order.append([int(val) for val in row])
route = []
for i in range(len(coords)):
    route.append(coords[visit_order[9999][i]])

route.append(route[0])

total_distance = 0
for i in range(1, len(route)):
    total_distance += np.linalg.norm(np.array(route[i]) - np.array(route[i - 1]))
    
x_vals, y_vals = zip(*route)

plt.figure(figsize=(6, 6))
plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
plt.scatter(*zip(*coords), color='r')
plt.title(f"TSP Route (Total Distance: {total_distance:.2f})")
plt.xlabel("X")
plt.ylabel("Y")

for idx, (x, y) in enumerate(coords):
    plt.text(x, y, f' {idx}', fontsize=12, color='black')

plt.grid(True)
plt.show()

print(f"Total Distance: {total_distance:.2f}")
