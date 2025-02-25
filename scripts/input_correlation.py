import os, sys

repo_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(repo_path)

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import torch
from old_src.models import Oja, BCM, SparseNet
from old_src.datasets import NatPatchDataset
from old_src.plotting import beeswarm, make_rf_display, save_figure
from old_src.files import get_figure_dir


if __name__ == "__main__":
    # Create a dataloader
    num_patches = 20000
    L = 10
    batch_size = 2000
    dataset = NatPatchDataset(num_patches, L)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    num_units = 144
    num_inputs = L**2
    oja = Oja(num_units, num_inputs, method="oja")
    bcm = BCM(num_units, num_inputs, y0=1.0, eps=0.001)
    snet = SparseNet(num_units, dataset.images[0].shape, method="FISTA")

    # Train
    lr = 1e-2
    optim_oja = torch.optim.SGD(oja.parameters(), lr=lr)
    optim_snet = torch.optim.SGD(snet.parameters(), lr=lr)
    for epoch in tqdm(range(20)):
        for batch in dataloader:
            data = batch.reshape(batch.size(0), -1)

            oja.zero_grad()
            out = oja(data)
            loss = oja.loss(data, out)
            loss.backward()
            optim_oja.step()
            oja.normalize_weights()

            out = bcm(data)
            bcm.update_weights(data, out, lr=lr * 0.1)
            bcm.normalize_weights()

            out = snet(data)[1].view(batch_size, -1)
            loss = (data - out).pow(2).sum()
            loss.backward()
            optim_snet.step()
            snet.zero_grad()
            snet.normalize_weights()

    figdir = get_figure_dir("sparsecoding")

    big_batch = dataloader.dataset.images.view(-1, L**2)
    corr_data = torch.corrcoef(big_batch.T)

    def rq(W, C):
        return torch.sum((C @ W) * W, dim=0) / torch.sum(W**2, dim=0)

    D, Q = torch.linalg.eigh(torch.cov(big_batch.T))
    idx_sort = torch.argsort(D, descending=True)
    D = D[idx_sort]
    Q = Q[:, idx_sort]

    snetw = snet.W.clone()

    nbins = 10
    ojarq = rq(oja.W, corr_data).detach().numpy() / num_inputs
    bcmrq = rq(bcm.W, corr_data).detach().numpy() / num_inputs
    snetrq = rq(snetw, corr_data).detach().numpy() / num_inputs
    xoja = beeswarm(ojarq, nbins)
    xbcm = beeswarm(bcmrq, nbins)
    xsnet = beeswarm(snetrq, nbins)

    names = ["Oja", "BCM", "SparseNet"]

    disp_buffer = 1
    num_to_show = 49
    ojarf = make_rf_display(
        oja.W.detach()[:, :num_to_show],
        disp_buffer=disp_buffer,
        flip_sign=False,
        background_value=1.0,
    )
    bcmrf = make_rf_display(
        bcm.W.detach()[:, :num_to_show],
        disp_buffer=disp_buffer,
        flip_sign=False,
        background_value=1.0,
    )
    snetrf = make_rf_display(
        snetw[:, :num_to_show],
        disp_buffer=disp_buffer,
        flip_sign=False,
        background_value=1.0,
    )

    plt.rcParams.update({"font.size": 16})

    # Create figure and GridSpec
    figdim = 2
    fig = plt.figure(figsize=(figdim * 6, figdim * 3.75))  # , layout="constrained")
    gs = GridSpec(4, 3, height_ratios=[2, 2, 1, 1], hspace=0.1, wspace=0.1)

    # Receptive Fields
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax3 = fig.add_subplot(gs[0:2, 2])

    # Eigenvector Composition
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[2, 2])

    # Average Eigenvector Composition
    ax7 = fig.add_subplot(gs[3, 0])
    ax8 = fig.add_subplot(gs[3, 1])
    ax9 = fig.add_subplot(gs[3, 2])

    # Plot data
    ax1.imshow(ojarf, cmap="gray", interpolation="none")
    ax2.imshow(bcmrf, cmap="gray", interpolation="none")
    ax3.imshow(snetrf, cmap="gray", interpolation="none")
    extent = [-2, Q.shape[1] + 1, 0, num_units]
    ax4.imshow(
        torch.abs(Q.T @ oja.W.detach().numpy()).T,
        vmin=0,
        vmax=1,
        cmap="pink",
        aspect="auto",
        extent=extent,
    )
    ax5.imshow(
        torch.abs(Q.T @ bcm.W.detach().numpy()).T,
        vmin=0,
        vmax=1,
        cmap="pink",
        aspect="auto",
        extent=extent,
    )
    ax6.imshow(
        torch.abs(Q.T @ snetw.numpy()).T,
        vmin=0,
        vmax=1,
        cmap="pink",
        aspect="auto",
        extent=extent,
    )
    ax7.plot(torch.mean(torch.abs(Q.T @ oja.W.detach().numpy()).T, dim=0), color="k")
    ax8.plot(torch.mean(torch.abs(Q.T @ bcm.W.detach().numpy()).T, dim=0), color="k")
    ax9.plot(torch.mean(torch.abs(Q.T @ snetw.numpy()).T, dim=0), color="k")

    # Format plots
    for ax, name in zip([ax1, ax2, ax3], names):
        ax.axis("off")
        ax.set_title(f"RFs - {name}")
    for ax in [ax4, ax5, ax6]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("E-Vec Composition")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    ax4.set_ylabel("Unit")
    for ax in [ax7, ax8, ax9]:
        ax.set_xlabel("Eigenvector")
        ax.set_xlim(-2, Q.shape[1] + 1)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax7.set_yticks([0, 1], labels=["0", "1"])
    ax7.set_ylabel("Contribution")
    ax8.spines["left"].set_visible(False)
    ax9.spines["left"].set_visible(False)
    ax8.set_yticks([])
    ax9.set_yticks([])
    plt.show()
    save_figure(fig, figdir / "RFs")

    fig = plt.figure(figsize=(3, 6.2), layout="constrained")
    for i, (x, y, name) in enumerate(
        zip([xoja, xbcm, xsnet], [ojarq, bcmrq, snetrq], names)
    ):
        plt.scatter(i + x / 3, y, c="k", alpha=0.3)
    plt.xticks(range(len(names)), names)
    plt.xticks(rotation=45)
    plt.xlim(-0.8, len(names) - 0.2)
    plt.ylabel("Average Input Correlation")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.show()
    save_figure(fig, figdir / "RQ-Corr-Norm")
