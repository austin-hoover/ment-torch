# https://github.com/AdamCobb/hamiltorch
from enum import Enum

from numpy import pi
import torch
import torch.nn as nn

from . import hmc_utils as util


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


def gibbs(params: torch.Tensor, mass: torch.Tensor | list) -> torch.Tensor:
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
    jitter: float = 0.01,
    normalizing_const: float = 1.0,
    softabs_const: float = 1e6,
    explicit_binding_const: float = 100.0,
    fixed_point_threshold: float = 1e-20,
    fixed_point_max_iterations: int = 6,
    jitter_max_tries: int = 10,
    inv_mass: torch.Tensor = None,
    sampler=Sampler.HMC,
    integrator=Integrator.IMPLICIT,
    metric=Metric.HESSIAN,
    store_on_gpu=True,
    debug=False,
    pass_grad=None,
):
    """This is a rather large function that contains all the various integration schemes
    used for HMC. Broadly speaking, it takes in the parameters and momentum and propose a
    new set of parameters and momentum. This is a key part of hamiltorch as it covers
    multiple integration schemes.

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
    jitter : float
        Jitter is often added to the diagonal to the metric tensor to ensure it can
        be inverted.
        `jitter` is a float corresponding to scale of random draws from a uniform distribution.
    normalizing_const : float
        This constant is currently set to 1.0 and might be removed in future versions as it
        plays no immediate role.
    softabs_const : float
        Controls the "filtering" strength of the negative eigenvalues. Large values -> absolute
         value. See Betancourt 2013.
    explicit_binding_const : float
        Only relevant to Explicit RMHMC. Corresponds to the binding term in Cobb et al. 2019.
    fixed_point_threshold : float
        Only relevant for Implicit RMHMC. Sets the convergence threshold for 'breaking out'
        of the while loop for the generalised leapfrog.
    fixed_point_max_iterations : int
        Only relevant for Implicit RMHMC. Limits the number of fixed point iterations in the
         generalised leapforg.
    jitter_max_tries : float
        Only relevant for RMHMC. Number of attempts at resampling the jitter for the Fisher
        Information before raising a LogProbError.
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
    metric : Metric
        Determines the metric to be used for RMHMC. E.g. default is the Hessian
        hamiltorch.Metric.HESSIAN.
    store_on_gpu : bool
        Option that determines whether to keep samples in GPU memory. It runs fast when set
        to TRUE but may run out of memory unless set to FALSE.
    debug : int
        This is useful for checking how many iterations RMHMC takes to converge. Set to zero
        for no print statements.
    pass_grad : None or torch.tensor or callable.
        If set to a torch.tensor, it is used as the gradient  shape: (D,), where D is the
        number of parameters of the model. If set to callable, it is a function to be called
        instead of evaluating the gradient directly using autograd. None is default and
        means autograd is used.

    Returns
    -------
    ret_params : list
        List of parameters collected in the trajectory. Note that explicit RMHMC returns
        a copy of two lists.
    ret_momenta : list
        List of momentum collected in the trajectory. Note that explicit RMHMC returns
        a copy of two lists.
    """
    params = params.clone()
    momentum = momentum.clone()
    # TodO detach graph when storing ret_params for memory saving
    if (
        sampler == Sampler.HMC
        and integrator != Integrator.SPLITTING
        and integrator != Integrator.SPLITTING_RAND
        and integrator != Integrator.SPLITTING_KMID
    ):

        def params_grad(p):
            p = p.detach().requires_grad_()
            log_prob = log_prob_func(p)
            # log_prob.backward()
            p = collect_gradients(log_prob, p, pass_grad)
            # print(p.grad.std())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return p.grad

        ret_params = []
        ret_momenta = []
        momentum += 0.5 * step_size * params_grad(params)
        for n in range(steps):
            if inv_mass is None:
                params = params + step_size * momentum  # /normalizing_const
            else:
                # Assum G is diag here so 1/Mass = G inverse
                if type(inv_mass) is list:
                    i = 0
                    for block in inv_mass:
                        it = block[0].shape[0]
                        params[i : it + i] = params[
                            i : it + i
                        ] + step_size * torch.matmul(
                            block, momentum[i : it + i].view(-1, 1)
                        ).view(
                            -1
                        )  # /normalizing_const
                        i += it
                elif len(inv_mass.shape) == 2:
                    params = params + step_size * torch.matmul(
                        inv_mass, momentum.view(-1, 1)
                    ).view(
                        -1
                    )  # /normalizing_const
                else:
                    params = (
                        params + step_size * inv_mass * momentum
                    )  # /normalizing_const
            p_grad = params_grad(params)
            momentum += step_size * p_grad
            ret_params.append(params.clone())
            ret_momenta.append(momentum.clone())
        # only need last for Hamiltoninian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
        ret_momenta[-1] = ret_momenta[-1] - 0.5 * step_size * p_grad.clone()
        # import pdb; pdb.set_trace()
        return ret_params, ret_momenta
    elif sampler == Sampler.RMHMC and (
        integrator == Integrator.IMPLICIT or integrator == Integrator.S3
    ):
        if integrator is not Integrator.S3:
            ham_func = None
            # Else we are doing semi sep and need auxiliary for Riemann version.
        if pass_grad is not None:
            raise RuntimeError(
                "Passing user-determined gradients not implemented for RMHMC"
            )

        def fixed_point_momentum(params, momentum):
            momentum_old = momentum.clone()
            # print('s')
            for i in range(fixed_point_max_iterations):
                momentum_prev = momentum.clone()
                params = params.detach().requires_grad_()
                ham = hamiltonian(
                    params,
                    momentum,
                    log_prob_func,
                    jitter=jitter,
                    softabs_const=softabs_const,
                    normalizing_const=normalizing_const,
                    ham_func=ham_func,
                    sampler=sampler,
                    integrator=integrator,
                    metric=metric,
                )
                params = collect_gradients(ham, params)

                # draw the jitter on the diagonal of Fisher again (probably a better place to do this)
                tries = 0
                while util.has_nan_or_inf(params.grad):
                    params = params.detach().requires_grad_()
                    ham = hamiltonian(
                        params,
                        momentum,
                        log_prob_func,
                        jitter=jitter,
                        softabs_const=softabs_const,
                        normalizing_const=normalizing_const,
                        ham_func=ham_func,
                        sampler=sampler,
                        integrator=integrator,
                        metric=metric,
                    )
                    params = collect_gradients(ham, params)
                    tries += 1
                    if tries > jitter_max_tries:
                        print(
                            "Warning: reached jitter_max_tries {}".format(
                                jitter_max_tries
                            )
                        )
                        # import pdb; pdb.set_trace()
                        raise util.LogProbError()
                        # import pdb; pdb.set_trace()
                        # break

                momentum = momentum_old - 0.5 * step_size * params.grad
                momenta_diff = torch.max((momentum_prev - momentum) ** 2)
                if momenta_diff < fixed_point_threshold:
                    break
            if debug == 1:
                print(
                    "Converged (momentum), iterations: {}, momenta_diff: {}".format(
                        i, momenta_diff
                    )
                )
            return momentum

        def fixed_point_params(params, momentum):
            params_old = params.clone()
            momentum = momentum.detach().requires_grad_()
            ham = hamiltonian(
                params,
                momentum,
                log_prob_func,
                jitter=jitter,
                softabs_const=softabs_const,
                normalizing_const=normalizing_const,
                ham_func=ham_func,
                sampler=sampler,
                integrator=integrator,
                metric=metric,
            )
            momentum = collect_gradients(ham, momentum)
            momentum_grad_old = momentum.grad.clone()
            for i in range(fixed_point_max_iterations):
                params_prev = params.clone()
                momentum = momentum.detach().requires_grad_()
                ham = hamiltonian(
                    params,
                    momentum,
                    log_prob_func,
                    jitter=jitter,
                    softabs_const=softabs_const,
                    normalizing_const=normalizing_const,
                    ham_func=ham_func,
                    sampler=sampler,
                    integrator=integrator,
                    metric=metric,
                )
                momentum = collect_gradients(
                    ham, momentum
                )  # collect_gradients(ham, params)
                params = (
                    params_old
                    + 0.5 * step_size * momentum.grad
                    + 0.5 * step_size * momentum_grad_old
                )
                params_diff = torch.max((params_prev - params) ** 2)
                if params_diff < fixed_point_threshold:
                    break
            if debug == 1:
                print(
                    "Converged (params), iterations: {}, params_diff: {}".format(
                        i, params_diff
                    )
                )
            return params

        ret_params = []
        ret_momenta = []
        for n in range(steps):
            # import pdb; pdb.set_trace()
            momentum = fixed_point_momentum(params, momentum)
            params = fixed_point_params(params, momentum)

            params = params.detach().requires_grad_()
            ham = hamiltonian(
                params,
                momentum,
                log_prob_func,
                jitter=jitter,
                softabs_const=softabs_const,
                normalizing_const=normalizing_const,
                ham_func=ham_func,
                sampler=sampler,
                integrator=integrator,
                metric=metric,
            )
            params = collect_gradients(ham, params)

            # draw the jitter on the diagonal of Fisher again (probably a better place to do this)
            tries = 0
            while util.has_nan_or_inf(params.grad):
                params = params.detach().requires_grad_()
                ham = hamiltonian(
                    params,
                    momentum,
                    log_prob_func,
                    jitter=jitter,
                    softabs_const=softabs_const,
                    normalizing_const=normalizing_const,
                    ham_func=ham_func,
                    sampler=sampler,
                    integrator=integrator,
                    metric=metric,
                )
                params = collect_gradients(ham, params)
                tries += 1
                if tries > jitter_max_tries:
                    print(
                        "Warning: reached jitter_max_tries {}".format(jitter_max_tries)
                    )
                    raise util.LogProbError()
                    # break
            momentum -= 0.5 * step_size * params.grad

            ret_params.append(params)
            ret_momenta.append(momentum)
        return ret_params, ret_momenta

    elif sampler == Sampler.RMHMC and integrator == Integrator.EXPLICIT:
        if pass_grad is not None:
            raise RuntimeError(
                "Passing user-determined gradients not implemented for RMHMC"
            )

        # During leapfrog define integrator as implict when passing into riemannian_hamiltonian
        leapfrog_hamiltonian_flag = Integrator.IMPLICIT

        def hamAB_grad_params(params, momentum):
            params = params.detach().requires_grad_()
            ham = hamiltonian(
                params,
                momentum.detach(),
                log_prob_func,
                jitter=jitter,
                normalizing_const=normalizing_const,
                softabs_const=softabs_const,
                explicit_binding_const=explicit_binding_const,
                sampler=sampler,
                integrator=leapfrog_hamiltonian_flag,
                metric=metric,
            )
            params = collect_gradients(ham, params)

            # draw the jitter on the diagonal of Fisher again (probably a better place to do this)
            tries = 0
            while util.has_nan_or_inf(params.grad):
                # import pdb; pdb.set_trace()
                params = params.detach().requires_grad_()
                ham = hamiltonian(
                    params,
                    momentum.detach(),
                    log_prob_func,
                    jitter=jitter,
                    normalizing_const=normalizing_const,
                    softabs_const=softabs_const,
                    explicit_binding_const=explicit_binding_const,
                    sampler=sampler,
                    integrator=leapfrog_hamiltonian_flag,
                    metric=metric,
                )
                params = collect_gradients(ham, params)
                tries += 1
                if tries > jitter_max_tries:
                    print(
                        "Warning: reached jitter_max_tries {}".format(jitter_max_tries)
                    )
                    raise util.LogProbError()
                    # import pdb; pdb.set_trace()
                    # break

            return params.grad

        def hamAB_grad_momentum(params, momentum):
            momentum = momentum.detach().requires_grad_()
            params = params.detach().requires_grad_()
            # Can't detach p as we still need grad to do derivatives
            ham = hamiltonian(
                params,
                momentum,
                log_prob_func,
                jitter=jitter,
                normalizing_const=normalizing_const,
                softabs_const=softabs_const,
                explicit_binding_const=explicit_binding_const,
                sampler=sampler,
                integrator=leapfrog_hamiltonian_flag,
                metric=metric,
            )
            # import pdb; pdb.set_trace()
            momentum = collect_gradients(ham, momentum)
            return momentum.grad

        ret_params = []
        ret_momenta = []
        params_copy = params.clone()
        momentum_copy = momentum.clone()
        for n in range(steps):
            # \phi_{H_A}
            momentum = momentum - 0.5 * step_size * hamAB_grad_params(
                params, momentum_copy
            )
            params_copy = params_copy + 0.5 * step_size * hamAB_grad_momentum(
                params, momentum_copy
            )
            # \phi_{H_B}
            params = params + 0.5 * step_size * hamAB_grad_momentum(
                params_copy, momentum
            )
            momentum_copy = momentum_copy - 0.5 * step_size * hamAB_grad_params(
                params_copy, momentum
            )
            # \phi_{H_C}
            c = torch.cos(
                torch.FloatTensor([2 * explicit_binding_const * step_size])
            ).to(params.device)
            s = torch.sin(
                torch.FloatTensor([2 * explicit_binding_const * step_size])
            ).to(params.device)
            # params_add = params + params_copy
            # params_sub = params - params_copy
            # momentum_add = momentum + momentum_copy
            # momentum_sub = momentum - momentum_copy
            # ### CHECK IF THE VALUES ON THE RIGHT NEED TO BE THE OLD OR UPDATED ones
            # ### INSTINCT IS THAT USING UPDATED ONES IS BETTER
            # params = 0.5 * ((params_add) + c*(params_sub) + s*(momentum_sub))
            # momentum = 0.5 * ((momentum_add) - s*(params_sub) + c*(momentum_sub))
            # params_copy = 0.5 * ((params_add) - c*(params_sub) - s*(momentum_sub))
            # momentum_copy = 0.5 * ((momentum_add) + s*(params_sub) - c*(momentum_sub))
            params = 0.5 * (
                (params + params_copy)
                + c * (params - params_copy)
                + s * (momentum - momentum_copy)
            )
            momentum = 0.5 * (
                (momentum + momentum_copy)
                - s * (params - params_copy)
                + c * (momentum - momentum_copy)
            )
            params_copy = 0.5 * (
                (params + params_copy)
                - c * (params - params_copy)
                - s * (momentum - momentum_copy)
            )
            momentum_copy = 0.5 * (
                (momentum + momentum_copy)
                + s * (params - params_copy)
                - c * (momentum - momentum_copy)
            )

            # \phi_{H_B}
            params = params + 0.5 * step_size * hamAB_grad_momentum(
                params_copy, momentum
            )
            momentum_copy = momentum_copy - 0.5 * step_size * hamAB_grad_params(
                params_copy, momentum
            )
            # \phi_{H_A}
            momentum = momentum - 0.5 * step_size * hamAB_grad_params(
                params, momentum_copy
            )
            params_copy = params_copy + 0.5 * step_size * hamAB_grad_momentum(
                params, momentum_copy
            )

            ret_params.append(params.clone())
            ret_momenta.append(momentum.clone())
        return [ret_params, params_copy], [ret_momenta, momentum_copy]

    # PAGE 35 MCMC Using Hamiltonian dynamics (Neal 2011)
    elif sampler == Sampler.HMC and (
        integrator == Integrator.SPLITTING
        or integrator == Integrator.SPLITTING_RAND
        or Integrator.SPLITTING_KMID
    ):
        if type(log_prob_func) is not list:
            raise RuntimeError("For splitting log_prob_func must be list of functions")
        if pass_grad is not None:
            raise RuntimeError(
                "Passing user-determined gradients not implemented for splitting"
            )

        def params_grad(p, log_prob_func):
            # OLD:
            # p = p.detach().requires_grad_()
            # log_prob = log_prob_func(p)
            # # log_prob.backward()
            # p = collect_gradients(log_prob, p)
            # grad = p.grad
            # # For removing GPU memory for large data sets.
            # del p, log_prob
            # torch.cuda.empty_cache()

            p = p.detach().requires_grad_()
            log_prob = log_prob_func(p)
            # Need to check memory issues in collect_gradients
            grad = torch.autograd.grad(log_prob, p)[0]
            # For removing GPU memory for large data sets.
            del p, log_prob, log_prob_func
            torch.cuda.empty_cache()
            return grad

        params = (
            params.detach()
        )  # Detach as we do not need to remember graph until we pass into log_prob
        ret_params = []
        ret_momenta = []
        if integrator == Integrator.SPLITTING:
            M = len(log_prob_func)
            K_div = (M - 1) * 2
            if M == 1:
                raise RuntimeError(
                    "For symmetric splitting log_prob_func must be list of functions greater than length 1"
                )
            for n in range(steps):
                # Symmetric loop to ensure reversible
                for m in range(M):
                    # print('p ',n)
                    grad = params_grad(params, log_prob_func[m])
                    with torch.no_grad():
                        momentum += 0.5 * step_size * grad
                        del grad
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if m < M - 1:
                            # print('q ',n)
                            if inv_mass is None:
                                params += (
                                    step_size / K_div
                                ) * momentum  # /normalizing_const
                            else:
                                if type(inv_mass) is list:
                                    pass
                                # Assum G is diag here so 1/Mass = G inverse
                                elif len(inv_mass.shape) == 2:
                                    params += (step_size / K_div) * torch.matmul(
                                        inv_mass, momentum.view(-1, 1)
                                    ).view(
                                        -1
                                    )  # /normalizing_const
                                else:
                                    params += (
                                        (step_size / K_div) * inv_mass * momentum
                                    )  # /normalizing_const
                for m in reversed(range(M)):
                    # print('p ', n )
                    grad = params_grad(params, log_prob_func[m])
                    with torch.no_grad():
                        momentum += 0.5 * step_size * grad
                        del grad
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if m > 0:
                            # print('q ', n-1)
                            if inv_mass is None:
                                params += (
                                    step_size / K_div
                                ) * momentum  # /normalizing_const
                            else:
                                if type(inv_mass) is list:
                                    pass
                                # Assum G is diag here so 1/Mass = G inverse
                                elif len(inv_mass.shape) == 2:
                                    params += (step_size / K_div) * torch.matmul(
                                        inv_mass, momentum.view(-1, 1)
                                    ).view(
                                        -1
                                    )  # /normalizing_const
                                else:
                                    params += (
                                        (step_size / K_div) * inv_mass * momentum
                                    )  # /normalizing_const

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
                # "Labelling of subsets is randomised for each iteration"
                # idx = torch.randperm(M)
                for m in range(M):
                    # print('p ',n)
                    momentum += (
                        0.5 * step_size * params_grad(params, log_prob_func[idx[m]])
                    )
                    # print('q ',n)
                    if inv_mass is None:
                        params += (step_size / M) * momentum  # /normalizing_const
                    else:
                        if type(inv_mass) is list:
                            pass
                        # Assum G is diag here so 1/Mass = G inverse
                        elif len(inv_mass.shape) == 2:
                            params += (step_size / M) * torch.matmul(
                                inv_mass, momentum.view(-1, 1)
                            ).view(
                                -1
                            )  # /normalizing_const
                        else:
                            params += (
                                (step_size / M) * inv_mass * momentum
                            )  # /normalizing_const
                    momentum += (
                        0.5 * step_size * params_grad(params, log_prob_func[idx[m]])
                    )

                ret_params.append(params.clone())
                ret_momenta.append(momentum.clone())
            # import pdb; pdb.set_trace()

        elif integrator == Integrator.SPLITTING_KMID:
            M = len(log_prob_func)
            if M == 1:
                raise RuntimeError(
                    "For symmetric splitting log_prob_func must be list of functions greater than length 1"
                )
            for n in range(steps):
                # Symmetric loop to ensure reversible
                for m in range(M):
                    # print('p ',n)
                    momentum += 0.5 * step_size * params_grad(params, log_prob_func[m])

                if inv_mass is None:
                    params = params + (step_size) * momentum  # /normalizing_const
                else:
                    if type(inv_mass) is list:
                        pass
                    # Assum G is diag here so 1/Mass = G inverse
                    elif len(inv_mass.shape) == 2:
                        params = params + (step_size) * torch.matmul(
                            inv_mass, momentum.view(-1, 1)
                        ).view(
                            -1
                        )  # /normalizing_const
                    else:
                        params = (
                            params + (step_size) * inv_mass * momentum
                        )  # /normalizing_const

                for m in reversed(range(M)):
                    # print('p ', n )
                    momentum += 0.5 * step_size * params_grad(params, log_prob_func[m])

                ret_params.append(params.clone())
                ret_momenta.append(momentum.clone())

        return ret_params, ret_momenta

    else:
        raise NotImplementedError()


def acceptance(h_old, h_new):
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


# Adaptation p.15 No-U-Turn samplers Algo 5
def adaptation(rho, t, step_size_init, H_t, eps_bar, desired_accept_rate=0.8):
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
    # rho is current acceptance ratio
    # t is current iteration
    t = t + 1
    if util.has_nan_or_inf(torch.tensor([rho])):
        alpha = 0  # Acceptance rate is zero if nan.
    else:
        alpha = min(1.0, float(torch.exp(torch.FloatTensor([rho]))))
    mu = float(torch.log(10 * torch.FloatTensor([step_size_init])))
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    H_t = (1 - (1 / (t + t0))) * H_t + (1 / (t + t0)) * (desired_accept_rate - alpha)
    x_new = mu - (t**0.5) / gamma * H_t
    step_size = float(torch.exp(torch.FloatTensor([x_new])))
    x_new_bar = t**-kappa * x_new + (1 - t**-kappa) * torch.log(
        torch.FloatTensor([eps_bar])
    )
    eps_bar = float(torch.exp(x_new_bar))

    return step_size, eps_bar, H_t


def hamiltonian(
    params,
    momentum,
    log_prob_func,
    jitter=0.01,
    normalizing_const=1.0,
    softabs_const=1e6,
    explicit_binding_const=100,
    inv_mass=None,
    ham_func=None,
    sampler=Sampler.HMC,
    integrator=Integrator.EXPLICIT,
    metric=Metric.HESSIAN,
):
    """Computes the Hamiltonian as a function of the parameters and the momentum.

    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters.
    momentum : torch.tensor
        Flat vector of momentum, corresponding to the parameters: shape (D,), where D is the dimensionality of the parameters.
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.
    jitter : float
        Jitter is often added to the diagonal to the metric tensor to ensure it can be inverted.
        `jitter` is a float corresponding to scale of random draws from a uniform distribution.
    normalizing_const : float
        This constant is currently set to 1.0 and might be removed in future versions as it plays no immediate role.
    softabs_const : float
        Controls the "filtering" strength of the negative eigenvalues. Large values -> absolute value. See Betancourt 2013.
    explicit_binding_const : float
        Only relevant to Explicit RMHMC. Corresponds to the binding term in Cobb et al. 2019.
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.
    ham_func : type
        Only related to semi-separable HMC. This part of hamiltorch has not been fully integrated yet.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.HMC, Sampler.RMHMC, Sampler.HMC_NUTS}.
    integrator : Integrator
        Sets the type of integrator to be used for the leapfrog: Choice {Integrator.EXPLICIT, Integrator.IMPLICIT, Integrator.SPLITTING,
        Integrator.SPLITTING_RAND, Integrator.SPLITTING_KMID}.
    metric : Metric
        Determines the metric to be used for RMHMC. E.g. default is the Hessian hamiltorch.Metric.HESSIAN.

    Returns
    -------
    torch.tensor
        Returns the value of the Hamiltonian: shape (1,).

    """
    if type(log_prob_func) is not list:
        log_prob = log_prob_func(params)

        if util.has_nan_or_inf(log_prob):
            print("Invalid log_prob: {}, params: {}".format(log_prob, params))
            raise util.LogProbError()

    elif type(log_prob_func) is list:  # I.e. splitting!
        log_prob = 0
        for split_log_prob_func in log_prob_func:
            # Don't propogate gradients for saving  GPU memory usage (Sampler.HMC code does not explicitly calculate dH/dp etc...)
            with torch.no_grad():
                log_prob = log_prob + split_log_prob_func(params)

                if util.has_nan_or_inf(log_prob):
                    print(
                        "Invalid log_prob: {}, params: {}".format(log_prob, params)
                    )
                    raise util.LogProbError()

    potential = -log_prob  # /normalizing_const
    if inv_mass is None:
        kinetic = 0.5 * torch.dot(momentum, momentum)  # /normalizing_const
    else:
        if type(inv_mass) is list:
            i = 0
            kinetic = 0
            for block in inv_mass:
                it = block[0].shape[0]
                kinetic = kinetic + 0.5 * torch.matmul(
                    momentum[i : it + i].view(1, -1),
                    torch.matmul(block, momentum[i : it + i].view(-1, 1)),
                ).view(
                    -1
                )  # /normalizing_const
                i += it
        # Assum G is diag here so 1/Mass = G inverse
        elif len(inv_mass.shape) == 2:
            kinetic = 0.5 * torch.matmul(
                momentum.view(1, -1), torch.matmul(inv_mass, momentum.view(-1, 1))
            ).view(
                -1
            )  # /normalizing_const
        else:
            kinetic = 0.5 * torch.dot(
                momentum, inv_mass * momentum
            )  # /normalizing_const
    hamiltonian = potential + kinetic
    # hamiltonian = hamiltonian

    return hamiltonian


def sample(
    log_prob_func,
    params_init,
    num_samples=10,
    num_steps_per_sample=10,
    step_size=0.1,
    burn=0,
    jitter=None,
    inv_mass=None,
    normalizing_const=1.0,
    softabs_const=None,
    explicit_binding_const=100,
    fixed_point_threshold=1e-5,
    fixed_point_max_iterations=1000,
    jitter_max_tries=10,
    sampler=Sampler.HMC,
    integrator=Integrator.IMPLICIT,
    metric=Metric.HESSIAN,
    debug=False,
    desired_accept_rate=0.8,
    store_on_gpu=True,
    pass_grad=None,
    verbose=True,
):
    """Run Hamiltonian Monte Carlo.

    Parameters
    ----------
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
    jitter : float
        Jitter is often added to the diagonal to the metric tensor to ensure it can be
        inverted. `jitter` is a float corresponding to scale of random draws from a 
        uniform distribution.
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the 
        covariance of the parameter space (the scale we expect it to vary). Currently
        this can be set to either a diagonal matrix, via a torch tensor of shape (D,), 
        or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the inv_mass matrix as a list of blocks. Hope 
        to make that more efficient.
    normalizing_const : float
        This constant is currently set to 1.0 and might be removed in future versions 
        as it plays no immediate role.
    softabs_const : float
        Controls the "filtering" strength of the negative eigenvalues. Large values -> 
        absolute value. See Betancourt 2013.
    explicit_binding_const : float
        Only relevant to Explicit RMHMC. Corresponds to the binding term in Cobb et al.
        2019.
    fixed_point_threshold : float
        Only relevant for Implicit RMHMC. Sets the convergence threshold for 'breaking
        out' of the while loop for the generalised leapfrog.
    fixed_point_max_iterations : int
        Only relevant for Implicit RMHMC. Limits the number of fixed point iterations 
        in the generalised leapforg.
    jitter_max_tries : float
        Only relevant for RMHMC. Number of attempts at resampling the jitter for the 
        Fisher Information before raising a LogProbError.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.HMC, 
        Sampler.RMHMC, Sampler.HMC_NUTS}.
    integrator : Integrator
        Sets the type of integrator to be used for the leapfrog: Choice 
        {Integrator.EXPLICIT, Integrator.IMPLICIT, Integrator.SPLITTING,
        Integrator.SPLITTING_RAND, Integrator.SPLITTING_KMID}.
    metric : Metric
        Determines the metric to be used for RMHMC. E.g. default is the Hessian 
        hamiltorch.Metric.HESSIAN.
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

    # Invert mass matrix once (As mass is used in Gibbs resampling step)
    mass = None
    if inv_mass is not None:
        if type(inv_mass) is list:
            mass = []
            for block in inv_mass:
                mass.append(torch.inverse(block))
        # Assum G is diag here so 1/Mass = G inverse
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
    # if sampler == Sampler.HMC:
    if verbose:
        util.progress_bar_init(
            "Sampling ({}; {})".format(sampler, integrator), num_samples, "Samples"
        )
    for n in range(num_samples):
        if verbose:
            util.progress_bar_update(n)
        try:
            momentum = gibbs(
                params,
                sampler=sampler,
                log_prob_func=log_prob_func,
                jitter=jitter,
                normalizing_const=normalizing_const,
                softabs_const=softabs_const,
                metric=metric,
                mass=mass,
            )

            ham = hamiltonian(
                params,
                momentum,
                log_prob_func,
                jitter=jitter,
                softabs_const=softabs_const,
                explicit_binding_const=explicit_binding_const,
                normalizing_const=normalizing_const,
                sampler=sampler,
                integrator=integrator,
                metric=metric,
                inv_mass=inv_mass,
            )

            leapfrog_params, leapfrog_momenta = leapfrog(
                params,
                momentum,
                log_prob_func,
                sampler=sampler,
                integrator=integrator,
                steps=num_steps_per_sample,
                step_size=step_size,
                inv_mass=inv_mass,
                jitter=jitter,
                jitter_max_tries=jitter_max_tries,
                fixed_point_threshold=fixed_point_threshold,
                fixed_point_max_iterations=fixed_point_max_iterations,
                normalizing_const=normalizing_const,
                softabs_const=softabs_const,
                explicit_binding_const=explicit_binding_const,
                metric=metric,
                store_on_gpu=store_on_gpu,
                debug=debug,
                pass_grad=pass_grad,
            )
            if sampler == Sampler.RMHMC and integrator == Integrator.EXPLICIT:
                # Step required to remove bias by comparing to Hamiltonian that is not 
                # augmented:
                ham = ham / 2  # Original RMHMC

                params = leapfrog_params[0][-1].detach().requires_grad_()
                params_copy = leapfrog_params[-1].detach().requires_grad_()
                params_copy = params_copy.detach().requires_grad_()
                momentum = leapfrog_momenta[0][-1]
                momentum_copy = leapfrog_momenta[-1]

                leapfrog_params = leapfrog_params[0]
                leapfrog_momenta = leapfrog_momenta[0]

                # This is trying the new (unbiased) version:
                new_ham = rm_hamiltonian(
                    params,
                    momentum,
                    log_prob_func,
                    jitter,
                    normalizing_const,
                    softabs_const=softabs_const,
                    sampler=sampler,
                    integrator=integrator,
                    metric=metric,
                )  
            else:
                params = leapfrog_params[-1].to(device).detach().requires_grad_()
                momentum = leapfrog_momenta[-1].to(device)
                new_ham = hamiltonian(
                    params,
                    momentum,
                    log_prob_func,
                    jitter=jitter,
                    softabs_const=softabs_const,
                    explicit_binding_const=explicit_binding_const,
                    normalizing_const=normalizing_const,
                    sampler=sampler,
                    integrator=integrator,
                    metric=metric,
                    inv_mass=inv_mass,
                )

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
                        # Store samples on CPU
                        ret_params.append(leapfrog_params[-1].cpu())
                else:
                    param_burn_prev = leapfrog_params[-1].to(device).clone()
            else:
                num_rejected += 1
                if n > burn:
                    params = ret_params[-1].to(device)
                    # leapfrog_params = ret_params[-num_steps_per_sample:] ### Might want to remove grad as wastes memory
                    if store_on_gpu:
                        ret_params.append(ret_params[-1].to(device))
                    else:
                        # Store samples on CPU
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

            # if not store_on_gpu: # i.e. delete stuff left on GPU
            #     # This adds approximately 50% to runtime when using colab 'Tesla P100-PCIE-16GB'
            #     # but leaves no memory footprint on GPU after use.
            #     # Might need to check if variables exist as a log prob error could occur before they are assigned!
            #
            #     del momentum, leapfrog_params, leapfrog_momenta, ham, new_ham
            #     torch.cuda.empty_cache()

        except util.LogProbError:
            num_rejected += 1
            params = ret_params[-1].to(device)
            if n > burn:
                params = ret_params[-1].to(device)
                # leapfrog_params = ret_params[-num_steps_per_sample:] ### Might want to remove grad as wastes memory
                if store_on_gpu:
                    ret_params.append(ret_params[-1].to(device))
                else:
                    # Store samples on CPU
                    ret_params.append(ret_params[-1].cpu())
            else:
                params = param_burn_prev.clone()
            if debug == 1:
                print("REJECT")
            if NUTS and n <= burn:
                # print('hi')
                rho = float("nan")  # Acceptance rate = 0
                # print(rho)
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

            # var_names = ['momentum', 'leapfrog_params', 'leapfrog_momenta', 'ham', 'new_ham']
            # [util.gpu_check_delete(var, locals()) for var in var_names]
            # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    if verbose:
        util.progress_bar_end(
            "Acceptance Rate {:.2f}".format(1 - num_rejected / num_samples)
        )  # need to adapt for burn
    if NUTS and debug == 2:
        return list(map(lambda t: t.detach(), ret_params)), step_size
    elif debug == 2:
        return (
            list(map(lambda t: t.detach(), ret_params)),
            1 - num_rejected / num_samples,
        )
    else:
        return list(map(lambda t: t.detach(), ret_params))
