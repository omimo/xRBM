import numpy as np

def plot_weight(fig, W, cmap='gray'):    
    ax = fig.add_subplot(121)
    ax.imshow(W, interpolation='nearest', aspect='auto', cmap=cmap)
    ax = fig.add_subplot(122)
    ax.hist(W)

    return ax


def plot_bias(fig, bias, cmap='gray'):
    ax = fig.add_subplot(121)
    ax.imshow(bias.T, interpolation='nearest', aspect='auto', cmap=cmap)
    # ax.colorbar()
    ax = fig.add_subplot(122)
    ax.hist(bias.T)


def create_2d_filters_grid(W, filter_shape, grid_size, grid_gap=(0,0)):
    out_shape = (grid_size[0]*(filter_shape[0]+grid_gap[0]), 
                 grid_size[1]*(filter_shape[1]+grid_gap[1]))

    out_img = np.zeros(out_shape)
    r = 0
    c = 0

    def scaleme(ww):
        return (ww - ww.min())/(ww.max() - ww.min())

    for w in W:
        im = scaleme(w.reshape(filter_shape))
        out_img[r:r+filter_shape[0], c:c+filter_shape[1]] = im

        c += filter_shape[1]+grid_gap[1]

        if c>=out_shape[1]:
            c = 0
            r = r + filter_shape[0] + 1
    
    return out_img
