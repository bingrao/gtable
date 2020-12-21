import torch.optim as optim


def build_torch_optimizer(model, opt):
    """Builds the PyTorch optimizer.

    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf
    These values are also used by other established implementations,
    e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    https://keras.io/optimizers/
    Recently there are slightly different values used in the paper
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    was used there however, beta2=0.999 is still arguably the more
    established value, so we use that here as well

    Args:
      model: The model to optimize.
      opt: The dictionary of options.

    Returns:
      A ``torch.optim.Optimizer`` instance.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if opt.optim == 'sgd':
        optimizer = optim.SGD(params, lr=opt.learning_rate)
    elif opt.optim == 'adagrad':
        optimizer = optim.Adagrad(
            params,
            lr=opt.learning_rate,
            initial_accumulator_value=opt.adagrad_accumulator_init)
    elif opt.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=opt.learning_rate)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(
            params,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=opt.learning_rate_decay,
            eps=1e-9)
    else:
        raise ValueError('Invalid optimizer type: ' + opt.optim)

    return optimizer
