import random
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import k_means
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from color_distillation.loss.similarity_preserving import BatchSimLoss, ce_loss, matching_ce_loss


class Dataset2Dpoints(Dataset):
    def __init__(self, N, component_centers):
        self.num_clusters = len(component_centers)
        self.points, self.labels = make_blobs(n_samples=N, centers=component_centers, cluster_std=0.4)
        self.points = torch.tensor(self.points, dtype=torch.float)
        self.labels = torch.tensor(self.labels, dtype=torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.points[item], self.labels[item]


def plot_decision_boundary(points, labels, model, title=None, cmap='Set3', fname=None):
    num_classes = len(np.unique(labels))

    # Set min and max values and give it some padding
    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
    h = 0.01  # step size in the mesh

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

    # Predict the function value for the whole grid
    model.eval()
    with torch.no_grad():
        Z = model(grid)
    _, Z = torch.max(Z, dim=1)
    Z = Z.numpy().reshape(xx.shape)

    # Plot the contour and training examples
    cmap = plt.cm.get_cmap(cmap, num_classes)
    fig1, ax1 = plt.subplots(figsize=(2, 2))
    ax1.set_aspect('equal')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    ax1.contourf(xx, yy, Z, cmap=cmap, alpha=0.2)
    ax1.scatter(points[:, 0], points[:, 1], s=10, c=labels, cmap=cmap, alpha=0.8, )
    # plt.colorbar(ticks=np.arange(num_classes))

    if title is not None:
        ax1.set_title(title)
    if fname is not None:
        fig1.savefig(f'{fname}.png')
    fig1.show()


def train(epoch, model, dataloader, optimizer, loss_fn, conf_ratio=0.0, info_ratio=0.0):
    model.train()
    loss_avg = 0
    for batch_idx, (point, label) in enumerate(dataloader):
        kmeans_center, kmeans_label, _ = k_means(point, dataloader.dataset.num_clusters)
        kmeans_label = torch.tensor(kmeans_label, dtype=torch.int64)

        output = model(point)
        prob = F.softmax(output, dim=1)
        loss = loss_fn(prob, F.one_hot(kmeans_label, dataloader.dataset.num_clusters).float())
        entropy = lambda prob: (-prob * torch.log(prob + 1e-16)).sum(dim=1)

        conf_loss = entropy(prob).mean()
        info_loss = -entropy(prob.mean(dim=0, keepdims=True)).sum()
        loss = loss + conf_loss * conf_ratio + info_loss + info_ratio
        loss_avg += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'epoch: {epoch}, batch: {batch_idx}, loss: {loss.item():.3f}')

    return loss_avg / len(dataloader)


def hungarian_test(model, dataloader):
    model.eval()
    target_probs, probs = [], []
    for batch_idx, (point, label) in enumerate(dataloader):
        with torch.no_grad():
            output = model(point)

        target_probs.append(F.one_hot(label, dataloader.dataset.num_clusters).float())
        probs.append(F.softmax(output, dim=1))
    target_probs, probs = torch.cat(target_probs).numpy(), torch.cat(probs).numpy()

    # compute the negative dot product between A and B
    cost = -target_probs.T @ probs

    # solve the assignment problem using the Hungarian algorithm
    # row_ind and col_ind contain the indices of the matched elements
    row_ind, col_ind = linear_sum_assignment(cost)

    targets, estimates = col_ind[target_probs.argmax(axis=1)], probs.argmax(axis=1)

    acc = (targets == estimates).mean()
    return acc


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    N = 1000
    component_centers = ([-1, 1], [-1, -1], [1, 1], [1, -1])
    dataset = Dataset2Dpoints(N, component_centers)

    B = 128
    dataloader = DataLoader(dataset, B, shuffle=True)

    # Plot init seeds along side sample data
    # plt.figure(1)
    # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
    # for i in range(len(component_centers)):
    #     cluster_data = dataset.labels == i
    #     plt.scatter(dataset.points[cluster_data, 0], dataset.points[cluster_data, 1], c=colors[i], marker=".", s=10)
    # # plt.xticks([])
    # # plt.yticks([])
    # plt.show()

    classifier = nn.Linear(2, len(component_centers), bias=True)

    optimizer = optim.SGD(classifier.parameters(), lr=0.2)

    acc = hungarian_test(classifier, dataloader)
    plot_decision_boundary(dataset.points, dataset.labels, classifier,
                           title=f'init classifier, acc: {acc:.1%}', fname='init')

    for epoch in range(1, 10 + 1):
        loss = train(epoch, classifier, dataloader, optimizer, loss_fn=ce_loss)
        # loss = train(epoch, classifier, dataloader, optimizer, loss_fn=matching_ce_loss)
        # loss = train(epoch, classifier, dataloader, optimizer, loss_fn=BatchSimLoss(normalize=False), info_ratio=1)
        # loss = train(epoch, classifier, dataloader, optimizer, loss_fn=BatchSimLoss(normalize=True), info_ratio=1)

        acc = hungarian_test(classifier, dataloader)
        plot_decision_boundary(dataset.points, dataset.labels, classifier,
                               title=f'epoch: {epoch}, acc: {acc:.1%}', fname=f'epoch{epoch:02d}')

    acc = hungarian_test(classifier, dataloader)
    plot_decision_boundary(dataset.points, dataset.labels, classifier, title=f'final classifier, acc: {acc:.1%}')
