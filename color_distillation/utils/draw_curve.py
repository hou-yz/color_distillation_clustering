import matplotlib.pyplot as plt


def draw_curve(path, x_epoch, train_loss, train_prec, test_loss, test_prec):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="prec")
    ax0.plot(x_epoch, train_loss, 'bo-', label=f'train: {train_loss[-1]:.3f}')
    ax1.plot(x_epoch, train_prec, 'bo-', label=f'train: {train_prec[-1]:.3f}')
    ax0.plot(x_epoch, test_loss, 'ro-', label=f'test: {test_loss[-1]:.3f}')
    ax1.plot(x_epoch, test_prec, 'ro-', label=f'test: {test_prec[-1]:.3f}')

    ax0.legend()
    ax1.legend()
    fig.savefig(path)
    plt.close(fig)
