import csv
import os

import diptest
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

folder_path = "distribution"
os.makedirs(folder_path, exist_ok=True)
csv_file_path = 'variable_values.csv'
header = ['epoch', 'value','class']
drop_class=[]
small_class=[]
# Write the header to the CSV file (if the file doesn't exist)
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)

def distribution(classes_distance,epoch,types):

    num_bins = 50

    dip = [diptest.diptest(vectors_np.cpu().numpy()) for vectors_np in classes_distance]

    histogram = [np.histogram(vectors_np.cpu().numpy(), bins=num_bins, density=True) for vectors_np in classes_distance]

    histograms_torch = [torch.tensor(hist[0], dtype=torch.float32) for hist in histogram]
    bin_edges_torch =  [torch.tensor(hist[1], dtype=torch.float32) for hist in histogram]

    # Calculate the continuous density estimate for each dimension using the midpoint of each bin
    x_values = [0.5 * (bin_edges[:-1] + bin_edges[1:]) for bin_edges in bin_edges_torch]
    density_estimates = [hist / torch.sum(hist) for hist in histograms_torch]
    # colors = ['red', 'blue', 'green', 'purple']
    colors = ['purple', 'gold', 'teal', 'pink', 'lime', 'gray', 'navy']
    cmap = mcolors.ListedColormap(colors)

    def save_to_csv(file_path, data):
        with open(file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(data)



    values = ([t[1] for t in dip])
    indexlist = [index for index, value in enumerate(values) if value <= 0.05 and index not in drop_class]
    if indexlist:
        for i in indexlist:
            mini =  values[i]
            epoch_data = [epoch, f'{mini:.3f}', i]
            save_to_csv(csv_file_path, epoch_data)
            plt.plot(x_values[i], density_estimates[i], label=f'class {i}', linewidth=2.5)
            drop_class.append(i)
            small_class.append(i)
    else:
        mini = min(values)
        indexi = values.index(mini)
        values_copy = values[:]
        while indexi in small_class:
            values_copy.remove(mini)
            mini = min(values_copy)
            indexi = values.index(mini)
        small_class.append(indexi)
        epoch_data = [epoch, f'{mini:.3f}', indexi]
        save_to_csv(csv_file_path, epoch_data)

        plt.plot(x_values[indexi], density_estimates[indexi], label=f'class_smaller {indexi}', linewidth=2.5)
    # for i in range(100):
        # plt.plot(x_values[i], density_estimates[i],color=cmap(i) ,label=f'class {dip[i][1]}',linewidth=2.5)
        # plt.plot(x_values[i], density_estimates[i], label=f'class {dip[i][1]}', linewidth=2.5)
        # print()

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    path = os.path.join("distribution", types + str(epoch) + ".png")
    plt.savefig(path)
    plt.close()
    return None
