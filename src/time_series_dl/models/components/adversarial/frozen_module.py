from typing import cast

from torch import Tensor, nn
from torch.func import functional_call


class FrozenModule:
    """
    Calls a module with detached copies of its parameters, so gradients reach its inputs but not its parameters.

    Adversarial objectives give each sub-network the gradient of its own phase loss; computing a term through frozen
    sub-networks lets one combined loss, minimized by a single optimizer, route that term's gradient to the intended
    sub-network only.

    Parameters
    ----------
    module : nn.Module
        The module to call; its parameters and buffers are read at every call, so it stays in sync with training.
    """

    def __init__(self, module: nn.Module) -> None:
        self.module: nn.Module = module

    def __call__(self, *inputs: Tensor) -> Tensor:
        """
        Run the module without tracking gradients of its parameters.

        Parameters
        ----------
        *inputs : Tensor
            The positional inputs of the module's forward.

        Returns
        -------
        Tensor
            The module's output.
        """
        detached: dict[str, Tensor] = {name: parameter.detach() for name, parameter in self.module.named_parameters()}
        return cast(Tensor, functional_call(self.module, detached, inputs))
