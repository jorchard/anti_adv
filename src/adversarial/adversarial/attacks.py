""" Adversarial Attacks """

from adversarial.networks import MyNet


def fgsm(net: MyNet, x, t, eps=0.01, targ=False):
    """
    x_adv = FGSM(net, x, t, eps=0.01, targ=False)

    Performs the Fast Gradient Sign Method, perturbing each input by
    eps (in infinity norm) in an attempt to have it misclassified.

    args:
      net    PyTorch Module object
      x      DxI tensor of a batch of inputs
      t      tensor of D corresponding class indices
      eps    the maximum infinity-norm perturbation from the input
      targ   Boolean, indicating if the FGSM is targetted
               - if targ is False, then t is considered to be the true
                 class of the input, and FGSM will work to increase the cost
                 for that target
               - if targ is True, then t is considered to be the target
                 class for the perturbation, and FGSM will work to decrease the
                 cost of the output for that target class

    return:
      x_adv  tensor of a batch of adversarial inputs, the same size as x
    """

    # Set the sign of the gradient-sign update [!!]
    signed_eps = -eps if targ else eps
    # If targetted, signed_eps should be negative
    # If not targetted, signed_eps should be positive
    # Tell PyTorch that we want to track the gradient all the way down to the input
    x.requires_grad = True
    # Feedforward
    y = net(x)
    # Compute loss and propagate gradients
    loss = net.classifier_loss(y, t)
    net.zero_grad()
    loss.backward()
    # The no_grad() is not really necessary
    with torch.no_grad():
        # Add +/-eps to each pixel
        x_adv = x + signed_eps * torch.sign(x.grad)

    return x_adv