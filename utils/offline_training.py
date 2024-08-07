import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import offline_metrics
import config


def rrTrain(X, y, lbda=0.0):
    '''
    Translated from Sam Nason's Utility folder on SVN. Trains a ridge regression and puts the model in p. It assumes
    that the first dimension of X and y are time, the second dimension of X is neuron, and the second dimension of y is
    in the order of [pos, vel, acc, 1]. Here, pos, vel, and acc should be [n, m] matrices where n is the number of
    samples and m is the number of movement dimensions.
    Inputs:
        - X (ndarray):
                Input neural data, which is size [n, neu], where n is the number of samples and neu is the
                number of neurons. X should already include a column of ones, if desired.
        - y (ndarray):
                Input behavior which is size [n, d], where n is the numebr of samples and m is the number of
                state dimensions.
        - lbda (scalar, optional):
                Customized value for lambda, defaults to 0.

    Outputs:
        - p (dict):
                p['Theta'] contains the trained theta matrix
                p['Lambda'] contains the value used for lambda
    '''
    p = {}

    temp = np.linalg.lstsq(np.matmul(X.T, X) + lbda * np.eye(X.shape[1]), np.matmul(X.T, y))
    p['Theta'] = temp[0]
    p['Lambda'] = lbda
    return p


def rrPredict(X, p):
    '''
    Translated from Sam Nason's Utility folder on SVN. This function makes ridge regression predictions from the neural
    data in X based on the params in p.
    Inputs:
        - X (ndarray):
                Input neural data, which is size [n, neu], wherre n is the number of samples and neu is the number of
                neurons. X should already include a column of ones, if desired.
        - p (dict):
                Input RR parameters. Should be a dict with at least the following field: theta. Run rrTrain first on
                training data to get these parameters made in the proper format
    Outputs:
        - yhat (ndarray):
                A size [n, d] matrix of predctions, where n is the number of samples and d is the number of state
                dimensions (dim 1 of p theta).
    '''
    yhat = None
    if 'Theta' in p:
        yhat = np.matmul(X, p['Theta'])
    else:
        ValueError('P does not contain a field Theta')
    return yhat


def dsTrain(X, y, vel_idx=None, post_prob=0.5, alpha=0.01, lbda=4, initK=None, initMoveProb=None):
    '''
    Trains a dual-state decoder based on work by Sachs et al. 2016. Separates movements into movement and posture states
    by sorting velocities, trains separate RR models on each, and trains the LDA weight matrix used for later prediction
    Inputs:
        - X (ndarray):
                Input neural data, which is size [n, neu], wherre n is the number of samples and neu is the number of
                neurons. X should already include a column of ones (REQUIRED).
        - y (ndarray):
                Input behavior which is size [n, d], where n is the numebr of samples and m is the number of state
                dimensions.
        - vel_idx (int list, optional):
                Indices of velocity states. Default None, assumes all dimensions are velocity
        - post_prop (float, optional):
                Optionally set the desires proportion of separation for movement vs posture states. Otherwise, defaults
                to 50/50 split.
        - alpha (float, optional):
                Update rate for moving threshold
        - lbda (float, optional):
                'Steepness' of logistic function, default 4
        - initK (float, optional):
                What initial k to start with in decoding. Default None, in which it's estimated from training data
        - initMoveProb (float, optional):
                Initial probability of movement to use in the first timestep.
                Default None, which which just assume standard mean movement probability.
    Outputs:
        - p (dict):
                p['move_theta'] contains the movement model. p['post_theta'] contains the posture model. p['W'] contains
                the LDA weights. p['postprob'], p['alpha'], p['lbda'], p['initK'], p['initMoveProb'] contain the
                respective parameters.
    '''

    move_prob = 1 - post_prob
    if vel_idx is None:
        vel_idx = np.arange(y.shape[1]).astype(int)
    if initMoveProb is None:
        initMoveProb = move_prob

    # separate states
    v_mag = np.sqrt(np.sum(y[:, vel_idx] ** 2, axis=1))
    mag_idx = np.argsort(v_mag)

    split_boundary = np.round(mag_idx.shape[0] * post_prob).astype(int)
    post_idx = mag_idx[:split_boundary]
    move_idx = mag_idx[split_boundary:]

    ypost = y[post_idx, :]
    ymove = y[move_idx, :]
    Xpost = X[post_idx, :]
    Xmove = X[move_idx, :]

    # Train Models for Movement and Posture
    lbda_best = 0.001

    post_model = rrTrain(Xpost, ypost, lbda=lbda_best)
    move_model = rrTrain(Xmove, ymove, lbda=lbda_best)

    # Train LDA
    S = np.cov(X[:, 0:-1].T)
    meandiff = np.mean(Xmove[:, :-1], axis=0) - np.mean(Xpost[:, :-1], axis=0)
    W = np.linalg.lstsq(S, meandiff)[0].T

    if initK is None:
        initK = np.dot(W, move_prob * (np.mean(Xmove[:, -1], axis=0) + np.mean(Xpost[:, :-1], axis=0)).T)

    p = {'move_model': move_model, 'post_model': post_model, 'W': W, 'alpha': alpha, 'lbda': lbda, 'initK': initK,
         'initMoveProb': initMoveProb, 'postprob': post_prob}
    return p


def dsPredict(X, p):
    '''
    Generates dual-state decoder predictions using pretrained matrices.
    Inputs:
        - X (ndarray):
                Input neural data, which is size [n, neu], wherre n is the number of samples and neu is the number of
                neurons. X should already include a column of ones, if desired.
        - p (dict):
                p['move_theta'] contains the movement model. p['post_theta'] contains the posture model. p['W'] contains
                the LDA weights. p['postprob'], p['alpha'], p['lbda'], p['initK'], p['initMoveProb'] contain the
                respective parameters.
    Outputs:
        - yhat(ndarray):
                A size [n,d] matrix of predictions, where n is the number of samples and d is the number of state
                dimensions (dim 1 of move and post models).
    '''
    alpha = p['alpha']
    kprev = p['initK']
    avgmoveprob = p['initMoveProb']
    lbda = p['lbda']
    meanWindowSize = 200
    postprob = p['postprob']
    moveprob = 1 - postprob
    move_model = p['move_model']
    post_model = p['post_model']
    W = p['W']

    k = kprev
    pmh = np.zeros((X.shape[0], 1))

    for i in np.arange(X.shape[0]):
        k = kprev + alpha * (avgmoveprob - moveprob)
        pm = 1 / (1 + np.exp(-lbda * (np.dot(W, X[i, :-1]) - k)))
        pmh[i] = pm
        avgmoveprob = np.mean(pmh[np.maximum(0, i + 1 - meanWindowSize):i + 1])
        kprev = k
    # pdb.set_trace()
    movepred = rrPredict(X, move_model)
    postpred = rrPredict(X, post_model)

    yhat = pmh * movepred + (1 - pmh) * postpred
    return yhat, pmh


# Basic Fit Function for forward/backprop dependent models
def fit(epochs, model, opt, dl, val_dl, print_every=1, print_results=False, scaler_used=True):
    best_epoch = 0
    loss_fn = torch.nn.MSELoss()
    loss_history = []
    vloss_history = []
    itertotal = 0
    for epoch in range(epochs):
        loss_list = []
        for x, y in dl:
            model.train()
            # 1. Generate your predictions
            yh = model(x)

            # 2. Find Loss
            loss = loss_fn(yh, y)
            loss_list.append(loss.item())
            # 3. Calculate gradients with respect to weights/biases
            loss.backward()

            # 4. Adjust your weights
            opt.step()

            # 5. Reset the gradients to zero
            opt.zero_grad()
            itertotal = itertotal + 1
        # Occasional progress update
        if print_results and ((epoch % print_every == 0) or (epoch == epochs - 1)):
            for x2, y2 in val_dl:
                with torch.no_grad():
                    model.eval()
                    if scaler_used:
                        scale_ds = BasicDataset(x2, y2)
                        scale_dl = DataLoader(scale_ds, batch_size=len(scale_ds))
                        scaler = generate_output_scaler(model, scale_dl, num_outputs=y2.shape[1], verbose=False)
                        yh = scaler.scale(model(x2))
                    else:
                        yh = model(x2)

                    val_loss = loss_fn(yh.cpu(), y2.cpu())
                    # val_cc = offline_metrics.corrcoef(yh.cpu(), y2.cpu())

                    print('Epoch [{}/{}], iter {} Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epochs - 1,
                                                                                                itertotal, loss.item(),
                                                                                                val_loss.item()))
                    # print('Validation Correlation: {:.4f}, {:.4f}'.format(val_cc[0], val_cc[1]))
                    train_loss = np.mean(np.asarray(loss_list))
                    loss_history.append(train_loss)
                    vloss_history.append(val_loss.item())
    return loss_history, vloss_history


class BasicDataset(Dataset):
    '''
    Torch Dataset if your neural and behavioral data are already all set-up with history, etc. Just sets up the
    chans_states attributes and returning the sample as a dict of 'chans' and 'states'.
    '''

    def __init__(self, chans, states):
        self.chans_states = (chans, states)

    def __len__(self):
        return len(self.chans_states[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        chans = self.chans_states[0][idx, :]
        states = self.chans_states[1][idx, :]

        sample = {'states': states, 'chans': chans}
        return sample


def generate_output_scaler(model, loader, num_outputs=2, verbose=True, is_refit=False, refit_orig_scaler=None):
    """Returns a scaler object that scales the output of a decoder

    Args:
        model:      model
        loader:     dataloader
        num_outputs:  how many outputs (2)
        is_refit:   if this is for a refit model
        refit_orig_scaler:  if refit, this is the scaler used in the refit training run.

    Returns:
        scaler: An OutputScaler object that takes returns scaled version of input data. If refit, this is the
                composition of the original and new scalers.
    """

    gains = None
    biases = None

    # fit constants using regression
    theta = calcGainRR(loader, model, idpt=True, verbose=verbose)
    # theta = [[w_x,   0]
    #          [0,   w_y]
    #          [b_x, b_y]]
    gains = np.zeros((1, num_outputs))
    biases = np.zeros((1, num_outputs))
    for i in range(num_outputs):
        gains[0, i] = theta[i, i]
        biases[0, i] = theta[num_outputs, i]

    # For refit, we apply both the old and new scalers:     y = m1*(m2x+b2) + b1
    if is_refit:
        gains = refit_orig_scaler.gains * gains
        biases = (refit_orig_scaler.gains * biases) + refit_orig_scaler.biases

    return OutputScaler(gains, biases)


def calcGainRR(loader, model, idpt=True, subtract_median=False, verbose=True):
    model.eval()  # set model to evaluation mode
    batches = len(list(loader))
    with torch.no_grad():
        for k1 in range(batches):  # TODO: handle multiple batches in loader
            temp = list(loader)
            x = temp[k1]['chans']
            # x = x[:, :, 0:ConvSize]
            y = temp[k1]['states']
            x = x.to(device=config.device, dtype=config.dtype)  # move to device, e.g. GPU
            y = y.to(device=config.device, dtype=config.dtype)
            yhat = model(x)
            if isinstance(yhat, tuple):
                # RNNs return y, h
                yhat = yhat[0]

            medians = []
            if subtract_median:
                for i in range(yhat.shape[1]):
                    medians.append(torch.median(yhat[:, i]))
                    yhat[:, i] = yhat[:, i] - torch.median(yhat[:, i])

            num_samps = yhat.shape[0]
            num_outputs = yhat.shape[1]
            yh_temp = torch.cat((yhat, torch.ones([num_samps, 1]).to(config.device)), dim=1)

            # JC notes:
            # yh_temp.shape[0] = num_samps
            # yh_temp.shape[1] = num_outputs+1

            if not idpt:
                # train theta normally (scaled velocities can depend on both input velocities)
                # Theta has the following form: [[w_xx, w_xy, b_x]  (actually transpose of this?)
                #                                [w_yx, w_yy, b_y]]
                theta = torch.mm(torch.mm(torch.pinverse(torch.mm(torch.t(yh_temp), yh_temp)), torch.t(yh_temp)), y)
                if verbose:
                    print(theta)
            else:
                # train ~special~ theta
                # (scaled velocities are indpendent of each other - this is the typical method)
                # Theta has the following form: [[w_x,   0]
                #                                [0,   w_y]
                #                                [b_x, b_y]]
                theta = torch.zeros((num_outputs + 1, num_outputs)).to(device=config.device, dtype=config.dtype)
                for i in range(num_outputs):
                    yhi = yh_temp[:, (i, -1)]
                    thetai = torch.matmul(torch.mm(torch.pinverse(torch.mm(torch.t(yhi), yhi)), torch.t(yhi)), y[:, i])
                    theta[i, i] = thetai[0]  # gain
                    theta[-1, i] = thetai[1]  # bias
                    if subtract_median:  # use the median as the bias
                        theta[-1, i] = -1 * medians[i]
                    if verbose:
                        print("Finger %d RR Calculated Gain, Offset: %.6f, %.6f" % (i, thetai[0], thetai[1]))
    return theta


class OutputScaler:

    def __init__(self, gains, biases, scaler_type=''):
        """An object to linearly scale data, like the output of a neural network

        Args:
            gains (1d np array):  [1,NumOutputs] array of gains
            biases (1d np array):           [1,NumOutputs] array of biases
            scaler_type (str, optional): 'regression' or 'peaks' or 'noscale', etc.
        """
        self.gains = gains
        self.biases = biases

    def scale(self, data):
        """
        data should be an numpy array/tensor of shape [N, NumOutputs]
        :param data:    np.ndarray or torch.Tensor, data to scale
        :return scaled_data np.ndarray or torch.Tensor, returns either according to what was input
        """

        # check if input is tensor or numpy
        isTensor = False
        if type(data) is torch.Tensor:
            isTensor = True
            data = data.cpu().detach().numpy()
        N = data.shape[0]

        # scale data
        scaled_data = np.tile(self.gains, (N, 1)) * data + np.tile(self.biases, (N, 1))

        # convert back to tensor if needed
        if isTensor:
            scaled_data = torch.from_numpy(scaled_data)

        return scaled_data

    def unscale(self, data):
        """Data should be an numpy array/tensor of shape [N, NumOutputs].
            Performs the inverse of the scale function (used in Refit)"""
        N = data.shape[0]
        # unscaled_data = (data / np.tile(self.gains, (N, 1))) - np.tile(self.biases, (N, 1))
        unscaled_data = (data - np.tile(self.biases, (N, 1))) / np.tile(self.gains, (N, 1))
        return unscaled_data
