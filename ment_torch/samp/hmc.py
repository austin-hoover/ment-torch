# https://github.com/AdamCobb/hamiltorch
from enum import Enum
from typing import Callable
from typing import Optional

from numpy import pi
import torch
import torch.nn as nn

from . import hmc_utils as utils


class Sampler(Enum):
    HMC = 1
    HMC_NUTS = 3


class Integrator(Enum):
    EXPLICIT = 1
    IMPLICIT = 2
    S3 = 3
    SPLITTING = 4
    SPLITTING_RAND = 5
    SPLITTING_KMID = 6


class Metric(Enum):
    HESSIAN = 1
    SOFTABS = 2
    JACOBIAN_DIAG = 3


def collect_gradients(log_prob: torch.Tensor, params: torch.Tensor, pass_grad=None) -> torch.Tensor:
    """Returns the parameters and the corresponding gradients (params.grad).

    Parameters
    ----------
    log_prob : torch.tensor
        Tensor shape (1,) which is a function of params (Can also be a tuple where log_prob[0] is the value to be differentiated).
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters .
    pass_grad : None or torch.tensor or callable.
        If set to a torch.tensor, it is used as the gradient  shape: (D,), where D is the number of parameters of the model. If set
        to callable, it is a function to be called instead of evaluating the gradient directly using autograd. None is default and
        means autograd is used.

    Returns
    -------
    torch.tensor
        The params, which is returned has the gradient attribute attached, i.e. params.grad.
    """
    if isinstance(log_prob, tuple):
        log_prob[0].backward()
        params_list = list(log_prob[1])
        params = torch.cat([p.flatten() for p in params_list])
        params.grad = torch.cat([p.grad.flatten() for p in params_list])
    elif pass_grad is not None:
        if callable(pass_grad):
            params.grad = pass_grad(params)
        else:
            params.grad = pass_grad
    else:
        params.grad = torch.autograd.grad(log_prob, params)[0]
    return params


def gibbs(params: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Performs the momentum resampling component of HMC.

    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters.
    mass : torch.tensor or list
        The mass matrix is related to the inverse covariance of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the mass matrix as a list of blocks. Hope to make that more efficient.

    Returns
    -------
    torch.tensor
        Returns the resampled momentum vector of shape (D,).
    """
    dist = None
    if mass is None:
        dist = torch.distributions.Normal(
            torch.zeros_like(params), torch.ones_like(params)
        )
    else:
        if type(mass) is list:
            # block wise mass list of blocks
            samples = torch.zeros_like(params)
            i = 0
            for block in mass:
                it = block[0].shape[0]
                dist = torch.distributions.MultivariateNormal(torch.zeros_like(block[0]), block)
                samples[i : it + i] = dist.sample()
                i += it
            return samples
        elif len(mass.shape) == 2:
            dist = torch.distributions.MultivariateNormal(
                torch.zeros_like(params), mass
            )
        elif len(mass.shape) == 1:
            dist = torch.distributions.Normal(torch.zeros_like(params), mass**0.5)
    return dist.sample()


def leapfrog(
    params: torch.Tensor,
    momentum: torch.Tensor,
    log_prob_func: Callable,
    steps: int = 10,
    step_size: float = 0.1,
    inv_mass: torch.Tensor = None,
    sampler=Sampler.HMC,
    integrator=Integrator.IMPLICIT,
    store_on_gpu=True,
    pass_grad=None,
):
    """Propose new set of parameters and momentum.

    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of
        the parameters.
    momentum : torch.tensor
        Flat vector of momentum, corresponding to the parameters: shape (D,), where D
        is the dimensionality of the parameters.
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of parameters
        that are being sampled.
    steps : int
        The number of steps to take per trajector (often referred to as L).
    step_size : float
        Size of each step to take when doing the numerical integration.
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance of
        the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix
        of shape (D,D). There is also the capability for some integration schemes to implement
         the inv_mass matrix as a list of blocks. Hope to make that more efficient.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.HMC,
        Sampler.RMHMC, Sampler.HMC_NUTS}.
    integrator : Integrator
        Sets the type of integrator to be used for the leapfrog: Choice {Integrator.EXPLICIT,
         Integrator.IMPLICIT, Integrator.SPLITTING, Integrator.SPLITTING_RAND,
         Integrator.SPLITTING_KMID}.
    store_on_gpu : bool
        Option that determines whether to keep samples in GPU memory. It runs fast when set
        to TRUE but may run out of memory unless set to FALSE.
    pass_grad : None or torch.tensor or callable.
        If set to a torch.tensor, it is used as the gradient  shape: (D,), where D is the
        number of parameters of the model. If set to callable, it is a function to be called
        instead of evaluating the gradient directly using autograd. None is default and
        means autograd is used.

    Returns
    -------
    ret_params : list
        List of parameters collected in the trajectory.
    ret_momenta : list
        List of momentum collected in the trajectory.
    """
    params = params.clone()
    momentum = momentum.clone()

    if (
        sampler == Sampler.HMC
        and integrator != Integrator.SPLITTING
        and integrator != Integrator.SPLITTING_RAND
        and integrator != Integrator.SPLITTING_KMID
    ):

        def params_grad(p):
            p = p.detach().requires_grad_()
            log_prob = log_prob_func(p)
            p = collect_gradients(log_prob, p, pass_grad)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return p.grad

        ret_params = []
        ret_momenta = []
        momentum += 0.5 * step_size * params_grad(params)
        for n in range(steps):
            if inv_mass is None:
                params = params + step_size * momentum
            else:
                # Assum G is diagonal here so 1/Mass = G inverse
                if type(inv_mass) is list:
                    i = 0
                    for block in inv_mass:
                        it = block[0].shape[0]
                        params[i : it + i] = params[i : it + i] + step_size * torch.matmul(block, momentum[i : it + i].view(-1, 1)).view(-1)
                        i += it
                elif len(inv_mass.shape) == 2:
                    params = params + step_size * torch.matmul(inv_mass, momentum.view(-1, 1)).view(-1)
                else:
                    params = params + step_size * inv_mass * momentum

            p_grad = params_grad(params)
            momentum += step_size * p_grad
            ret_params.append(params.clone())
            ret_momenta.append(momentum.clone())

        # only need last for Hamiltoninian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
        ret_momenta[-1] = ret_momenta[-1] - 0.5 * step_size * p_grad.clone()
        return ret_params, ret_momenta

    # PAGE 35 MCMC Using Hamiltonian dynamics (Neal 2011)
    elif sampler == Sampler.HMC and (
        integrator == Integrator.SPLITTING
        or integrator == Integrator.SPLITTING_RAND
        or Integrator.SPLITTING_KMID
    ):
        if type(log_prob_func) is not list:
            raise RuntimeError("For splitting log_prob_func must be list of functions")
        if pass_grad is not None:
            raise RuntimeError("Passing user-determined gradients not implemented for splitting")

        def params_grad(p, log_prob_func):
            p = p.detach().requires_grad_()
            log_prob = log_prob_func(p)
            # Need to check memory issues in collect_gradients
            grad = torch.autograd.grad(log_prob, p)[0]
            # For removing GPU memory for large data sets.
            del p, log_prob, log_prob_func
            torch.cuda.empty_cache()
            return grad

        # Detach as we do not need to remember graph until we pass into log_prob.
        params = params.detach()

        ret_params = []
        ret_momenta = []
        if integrator == Integrator.SPLITTING:
            M = len(log_prob_func)
            K_div = (M - 1) * 2
            if M == 1:
                raise RuntimeError("For symmetric splitting log_prob_func must be list of functions greater than length 1")
            for n in range(steps):
                # Symmetric loop to ensure reversible
                for m in range(M):
                    grad = params_grad(params, log_prob_func[m])
                    with torch.no_grad():
                        momentum += 0.5 * step_size * grad
                        del grad
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if m < M - 1:
                            if inv_mass is None:
                                params += (step_size / K_div) * momentum
                            else:
                                if type(inv_mass) is list:
                                    pass
                                # Assum G is diagonal here so 1/Mass = G inverse
                                elif len(inv_mass.shape) == 2:
                                    params += (step_size / K_div) * torch.matmul(inv_mass, momentum.view(-1, 1)).view(-1)
                                else:
                                    params += (step_size / K_div) * inv_mass * momentum
                for m in reversed(range(M)):
                    grad = params_grad(params, log_prob_func[m])
                    with torch.no_grad():
                        momentum += 0.5 * step_size * grad
                        del grad
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if m > 0:
                            if inv_mass is None:
                                params += (step_size / K_div) * momentum
                            else:
                                if type(inv_mass) is list:
                                    pass
                                # Assum G is diagonal here so 1/Mass = G inverse
                                elif len(inv_mass.shape) == 2:
                                    params += (step_size / K_div) * torch.matmul(inv_mass, momentum.view(-1, 1)).view(-1)
                                else:
                                    params += (step_size / K_div) * inv_mass * momentum

                if store_on_gpu:
                    ret_params.append(params.clone())
                    ret_momenta.append(momentum.clone())
                else:
                    ret_params.append(params.clone().cpu())
                    ret_momenta.append(momentum.clone().cpu())

        elif integrator == Integrator.SPLITTING_RAND:
            M = len(log_prob_func)
            idx = torch.randperm(M)
            for n in range(steps):
                # Labelling of subsets is randomised for each iteration
                for m in range(M):
                    momentum += 0.5 * step_size * params_grad(params, log_prob_func[idx[m]])
                    if inv_mass is None:
                        params += (step_size / M) * momentum
                    else:
                        if type(inv_mass) is list:
                            pass
                        # Assum G is diagonal here so 1/Mass = G inverse
                        elif len(inv_mass.shape) == 2:
                            params += (step_size / M) * torch.matmul(inv_mass, momentum.view(-1, 1)).view(-1)
                        else:
                            params += (step_size / M) * inv_mass * momentum
                    momentum += 0.5 * step_size * params_grad(params, log_prob_func[idx[m]])

                ret_params.append(params.clone())
                ret_momenta.append(momentum.clone())

        elif integrator == Integrator.SPLITTING_KMID:
            M = len(log_prob_func)
            if M == 1:
                raise RuntimeError("For symmetric splitting log_prob_func must be list of functions greater than length 1")
            for n in range(steps):
                # Symmetric loop to ensure reversible
                for m in range(M):
                    momentum += 0.5 * step_size * params_grad(params, log_prob_func[m])

                if inv_mass is None:
                    params = params + (step_size) * momentum
                else:
                    if type(inv_mass) is list:
                        pass
                    # Assum G is diagonal here so 1/Mass = G inverse
                    elif len(inv_mass.shape) == 2:
                        params = params + (step_size) * torch.matmul(inv_mass, momentum.view(-1, 1)).view(-1)
                    else:
                        params = params + (step_size) * inv_mass * momentum

                for m in reversed(range(M)):
                    momentum += 0.5 * step_size * params_grad(params, log_prob_func[m])

                ret_params.append(params.clone())
                ret_momenta.append(momentum.clone())

        return ret_params, ret_momenta

    else:
        raise NotImplementedError()


def acceptance(h_old: torch.Tensor, h_new: torch.Tensor) -> float:
    """Returns the log acceptance ratio for the Metroplis-Hastings step.

    Parameters
    ----------
    h_old : torch.tensor
        Previous value of Hamiltonian (1,).
    h_new : type
        New value of Hamiltonian (1,).

    Returns
    -------
    float
        Log acceptance ratio.
    """
    return float(-h_new + h_old)


def adaptation(rho: float, t: int, step_size_init: float, H_t: float, eps_bar: float, desired_accept_rate: float = 0.8) -> tuple:
    """No-U-Turn sampler adaptation of the step size. This follows Algo 5, p. 15 from Hoffman and Gelman 2011.

    Parameters
    ----------
    rho : float
        rho is current acceptance ratio.
    t : int
        Iteration.
    step_size_init : float
        Initial step size.
    H_t : float
        Current rolling H_t.
    eps_bar : type
        Current rolling step size update.
    desired_accept_rate : float
        The step size is adapted with the objective of a desired acceptance rate.

    Returns
    -------
    step_size : float
        Current step size to be used.
    eps_bar : float
        Current rolling step size update. Also at last iteration this is the final adapted step size.
    H_t : float
        Current rolling H_t to be passed at next iteration.
    """
    t = t + 1
    
    if utils.has_nan_or_inf(torch.tensor([rho])):
        alpha = 0.0  # Acceptance rate is zero if nan.
    else:
        alpha = min(1.0, float(torch.exp(torch.FloatTensor([rho]))))
        
    mu = float(torch.log(10 * torch.FloatTensor([step_size_init])))
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    H_t = (1 - (1 / (t + t0))) * H_t + (1 / (t + t0)) * (desired_accept_rate - alpha)
    x_new = mu - (t**0.5) / gamma * H_t
    step_size = float(torch.exp(torch.FloatTensor([x_new])))
    x_new_bar = t**-kappa * x_new + (1 - t**-kappa) * torch.log(torch.FloatTensor([eps_bar]))
    eps_bar = float(torch.exp(x_new_bar))

    return step_size, eps_bar, H_t


def hamiltonian(
    params: torch.Tensor,
    momentum: torch.Tensor,
    log_prob_func: Callable,
    inv_mass: torch.Tensor = None,
) -> torch.Tensor:
    """Computes the Hamiltonian as a function of the parameters and the momentum.

    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of 
        the parameters.
    momentum : torch.tensor
        Flat vector of momentum, corresponding to the parameters: shape (D,), where D 
        is the dimensionality of the parameters.
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of 
        parameters that are being sampled.
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance
        of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square 
        matrix of shape (D,D). There is also the capability for some integration schemes to
        implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.

    Returns
    -------
    torch.tensor
        Returns the value of the Hamiltonian: shape (1,).
    """
    if type(log_prob_func) is not list:
        log_prob = log_prob_func(params)

        if utils.has_nan_or_inf(log_prob):
            print("Invalid log_prob: {}, params: {}".format(log_prob, params))
            raise utils.LogProbError()

    elif type(log_prob_func) is list:  # I.e. splitting!
        log_prob = 0
        for split_log_prob_func in log_prob_func:
            # Don't propagate gradients for saving GPU memory usage (Sampler.HMC code does not explicitly calculate dH/dp etc...)
            with torch.no_grad():
                log_prob = log_prob + split_log_prob_func(params)
                if utils.has_nan_or_inf(log_prob):
                    print("Invalid log_prob: {}, params: {}".format(log_prob, params))
                    raise utils.LogProbError()

    potential = -log_prob
    if inv_mass is None:
        kinetic = 0.5 * torch.dot(momentum, momentum)
    else:
        if type(inv_mass) is list:
            i = 0
            kinetic = 0
            for block in inv_mass:
                it = block[0].shape[0]
                kinetic = kinetic + 0.5 * torch.matmul(momentum[i : it + i].view(1, -1), torch.matmul(block, momentum[i : it + i].view(-1, 1))).view(-1)
                i += it
        # Assume G is diagonal here so 1/Mass = G inverse
        elif len(inv_mass.shape) == 2:
            kinetic = 0.5 * torch.matmul(momentum.view(1, -1), torch.matmul(inv_mass, momentum.view(-1, 1))).view(-1)
        else:
            kinetic = 0.5 * torch.dot(momentum, inv_mass * momentum)
    hamiltonian = potential + kinetic
    return hamiltonian


def sample(
    log_prob_func: Callable,
    params_init: torch.Tensor,
    num_samples: int = 10,
    num_steps_per_sample: int = 10,
    step_size: float = 0.1,
    burn: int = 0,
    inv_mass=None,
    sampler=Sampler.HMC,
    integrator=Integrator.IMPLICIT,
    debug=False,
    desired_accept_rate=0.8,
    store_on_gpu=True,
    pass_grad=None,
    verbose=True,
):
    """Run Hamiltonian Monte Carlo.

    Parameters
    ----------
    log_prob_func : Callable
        A log_prob_func must take a 1-d vector of length equal to the number of
        parameters that are being sampled.
    params_init : torch.tensor
        Initialisation of the parameters. This is a vector corresponding to the 
        starting point of the sampler: shape: (D,), where D is the number of parameters
        of the model.
    num_samples : int
        Sets the number of samples corresponding to the number of momentum resampling 
        steps/the number of trajectories to sample.
    num_steps_per_sample : int
        The number of steps to take per trajector (often referred to as L).
    step_size : float
        Size of each step to take when doing the numerical integration.
    burn : int
        Number of samples to burn before collecting samples. Set to -1 for no burning
        of samples. This must be less than `num_samples` as `num_samples` subsumes 
        `burn`.
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the 
        covariance of the parameter space (the scale we expect it to vary). Currently
        this can be set to either a diagonal matrix, via a torch tensor of shape (D,), 
        or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the inv_mass matrix as a list of blocks. Hope 
        to make that more efficient.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.HMC, 
        Sampler.RMHMC, Sampler.HMC_NUTS}.
    integrator : Integrator
        Sets the type of integrator to be used for the leapfrog: Choice 
        {Integrator.EXPLICIT, Integrator.IMPLICIT, Integrator.SPLITTING,
        Integrator.SPLITTING_RAND, Integrator.SPLITTING_KMID}.
    debug : {0, 1, 2}
        Debug mode can take 3 options. Setting debug = 0 (default) allows the sampler 
        to run as normal. Setting debug = 1 prints both the old and new Hamiltonians
        per iteration, and also prints the convergence values when using the 
        generalised leapfrog (IMPLICIT RMHMC). Setting debug = 2, ensures an 
        additional float is returned corresponding to the acceptance rate or the 
        adapted step size (depending if NUTS is used.)
    desired_accept_rate : float
        Only relevant for NUTS. Sets the ideal acceptance rate that the NUTS will 
        converge to.
    store_on_gpu : bool
        Option that determines whether to keep samples in GPU memory. It runs fast when
        set to TRUE but may run out of memory unless set to FALSE.
    pass_grad : None or torch.tensor or callable.
        If set to a torch.tensor, it is used as the gradient  shape: (D,), where D is 
        the number of parameters of the model. If set to callable, it is a function to
        be called instead of evaluating the gradient directly using autograd. None is 
        default and means autograd is used.
    verbose : bool
        If set to true then do not display loading bar

    Returns
    -------
    param_samples : list of torch.tensor(s)
        A list of parameter samples. The full trajectory will be returned such that 
        selecting the proposed params requires indexing [1::L] to remove params_innit 
        and select the end of the trajectories.
    step_size : float, optional
        Only returned when debug = 2 and using NUTS. This is the final adapted step 
        size.
    acc_rate : float, optional
        Only returned when debug = 2 and not using NUTS. This is the acceptance rate.
    """
    device = params_init.device

    if params_init.dim() != 1:
        raise RuntimeError("params_init must be a 1d tensor.")

    if burn >= num_samples:
        raise RuntimeError("burn must be less than num_samples.")

    NUTS = False
    if sampler == Sampler.HMC_NUTS:
        if burn == 0:
            raise RuntimeError("burn must be greater than 0 for NUTS.")
        sampler = Sampler.HMC
        NUTS = True
        step_size_init = step_size
        H_t = 0.0
        eps_bar = 1.0

    mass = None
    if inv_mass is not None:
        if type(inv_mass) is list:
            mass = []
            for block in inv_mass:
                mass.append(torch.inverse(block))
        # Assum G is diagonal here so 1/Mass = G inverse
        elif len(inv_mass.shape) == 2:
            mass = torch.inverse(inv_mass)
        elif len(inv_mass.shape) == 1:
            mass = 1 / inv_mass

    params = params_init.clone().requires_grad_()
    param_burn_prev = params_init.clone()
    if not store_on_gpu:
        ret_params = [params.clone().detach().cpu()]
    else:
        ret_params = [params.clone()]

    num_rejected = 0

    if verbose:
        utils.progress_bar_init("Sampling ({}; {})".format(sampler, integrator), num_samples, "Samples")
    
    for n in range(num_samples):
        if verbose:
            utils.progress_bar_update(n)
        try:
            momentum = gibbs(params, mass=mass)
            ham = hamiltonian(params, momentum, log_prob_func, inv_mass=inv_mass)
            leapfrog_params, leapfrog_momenta = leapfrog(
                params,
                momentum,
                log_prob_func,
                sampler=sampler,
                integrator=integrator,
                steps=num_steps_per_sample,
                step_size=step_size,
                inv_mass=inv_mass,
                store_on_gpu=store_on_gpu,
                pass_grad=pass_grad,
            )

            params = leapfrog_params[-1].to(device).detach().requires_grad_()
            momentum = leapfrog_momenta[-1].to(device)
            new_ham = hamiltonian(params, momentum, log_prob_func, inv_mass=inv_mass)

            rho = min(0.0, acceptance(ham, new_ham))
            if debug == 1:
                message = ""
                message += "Step: {},".format(n)
                message += " Current Hamiltonian: {},".format(ham)
                message += " Proposed Hamiltonian: {}".format(new_ham)
                print(message)

            if rho >= torch.log(torch.rand(1)):
                if debug == 1:
                    print("Accept rho: {}".format(rho))
                if n > burn:
                    if store_on_gpu:
                        ret_params.append(leapfrog_params[-1])
                    else:
                        ret_params.append(leapfrog_params[-1].cpu())
                else:
                    param_burn_prev = leapfrog_params[-1].to(device).clone()
            else:
                num_rejected += 1
                if n > burn:
                    params = ret_params[-1].to(device)
                    if store_on_gpu:
                        ret_params.append(ret_params[-1].to(device))
                    else:
                        ret_params.append(ret_params[-1].cpu())
                else:
                    params = param_burn_prev.clone()
                if debug == 1:
                    print("REJECT")

            if NUTS and n <= burn:
                if n < burn:
                    step_size, eps_bar, H_t = adaptation(
                        rho,
                        n,
                        step_size_init,
                        H_t,
                        eps_bar,
                        desired_accept_rate=desired_accept_rate,
                    )
                if n == burn:
                    step_size = eps_bar
                    print("Final Adapted Step Size: ", step_size)

        except utils.LogProbError:
            num_rejected += 1
            params = ret_params[-1].to(device)
            if n > burn:
                params = ret_params[-1].to(device)
                if store_on_gpu:
                    ret_params.append(ret_params[-1].to(device))
                else:
                    ret_params.append(ret_params[-1].cpu())
            else:
                params = param_burn_prev.clone()
                
            if debug == 1:
                print("REJECT")
                
            if NUTS and n <= burn:
                rho = float("nan")
                step_size, eps_bar, H_t = adaptation(
                    rho,
                    n,
                    step_size_init,
                    H_t,
                    eps_bar,
                    desired_accept_rate=desired_accept_rate,
                )
            if NUTS and n == burn:
                step_size = eps_bar
                print("Final Adapted Step Size: ", step_size)

        if not store_on_gpu:  # i.e. delete stuff left on GPU
            # This adds approximately 50% to runtime when using colab 'Tesla P100-PCIE-16GB'
            # but leaves no memory footprint on GPU after use in normal HMC mode. (not split)
            # Might need to check if variables exist as a log prob error could occur before they are assigned!
            momentum = None
            leapfrog_params = None
            leapfrog_momenta = None
            ham = None
            new_ham = None

            del momentum, leapfrog_params, leapfrog_momenta, ham, new_ham
            torch.cuda.empty_cache()

    if verbose:
        utils.progress_bar_end("Acceptance Rate {:.2f}".format(1 - num_rejected / num_samples))

    if NUTS and debug == 2:
        return list(map(lambda t: t.detach(), ret_params)), step_size
    elif debug == 2:
        return (
            list(map(lambda t: t.detach(), ret_params)),
            1 - num_rejected / num_samples,
        )
    else:
        return list(map(lambda t: t.detach(), ret_params))
