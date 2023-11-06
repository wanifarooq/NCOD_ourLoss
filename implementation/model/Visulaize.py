# from PIL import Image
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

folder_path = "images"
os.makedirs(folder_path, exist_ok=True)
def visualization(vector,epoch,seed_emd,class_seed_flag):
    # class_seed_flag = class_seed_flag.float()
    vector = vector.view(3200, 512)
    seed_emd = seed_emd.view(100,512)
    vector = torch.cat((vector, seed_emd), dim=0)
    tsne = TSNE(n_components=3, perplexity=30, learning_rate=100, n_iter=2000)
    embedded_data = tsne.fit_transform(vector.cpu().numpy())


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define markers and colors for each class
    # markers = ['o', 's', '^', 'v', 'd', 'p', '*', 'h', 'x', '8']
    markers = ['o', 's', '^', 'v', 'd', 'p', '*', 'h', 'x', '+', '>', '<', 'D', '1']
    colors = ['red', 'blue', 'green','purple']
    # colors = ['purple', 'gold', 'teal', 'pink', 'lime', 'gray', 'navy']
    cmap = mcolors.ListedColormap(colors)
    # Plot data points with class labels
    for i in range(4):
        k= 32*i
        class_data = embedded_data[k:k+16]
        ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], marker=markers[i], color=cmap(i),alpha=0.8, s=150,
                   label=f'Class {i}')
        class_data = embedded_data[k+16:k + 32]
        ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], marker=markers[i+4], color=cmap(i),alpha=0.8, s=150,
                   label=f'Class {i}')
    seed_emd=embedded_data[3200:3204]
    for x,i in zip(seed_emd,range(4)):
        if class_seed_flag[i]:
            mark =i
        else:
            mark = i+4
        ax.scatter(x[0], x[1], x[2], marker=markers[mark], color=cmap(i), alpha=0.8,
                   s=300,
                   label=f'Class {i}')
    ax.set_title("t-SNE 3D Visualization with Class Labels")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend().remove()
    # plt.show()
    image_path = os.path.join("images", "tsne_visualization"+str(epoch)+".png")
    plt.savefig(image_path)
    plt.close(fig)

    return None