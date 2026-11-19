from collections.abc import Sequence


class WeightsMismatchError(ValueError):
    """
    Raised when published weights do not fit a network exactly.

    Parameters
    ----------
    missing : Sequence[str]
        Names of network parameters and buffers that the weights do not provide.
    unexpected : Sequence[str]
        Names, after mapping, of weights that the network does not have.
    mismatched : Sequence[str]
        Descriptions of weights whose shape differs from the network's.
    """

    def __init__(self, missing: Sequence[str], unexpected: Sequence[str], mismatched: Sequence[str]) -> None:
        self.missing: tuple[str, ...] = tuple(missing)
        self.unexpected: tuple[str, ...] = tuple(unexpected)
        self.mismatched: tuple[str, ...] = tuple(mismatched)
        parts: list[str] = [
            f"{label}: {', '.join(names)}"
            for label, names in (
                ("missing", self.missing),
                ("unexpected", self.unexpected),
                ("shape mismatch", self.mismatched),
            )
            if names
        ]
        super().__init__("published weights do not fit the network; " + "; ".join(parts))
