from enum import Enum


class MinimalRecurrentCell(Enum):
    """
    Minimal recurrent cell of Feng et al. (2024), whose gates read only the current input.

    Attributes
    ----------
    GRU
        minGRU: h_t = (1 - z_t) * h_{t-1} + z_t * h~_t with an update gate z_t.
    LSTM
        minLSTM: h_t = f'_t * h_{t-1} + i'_t * h~_t with forget and input gates normalized to sum to one.
    """

    GRU = "gru"
    LSTM = "lstm"

    @property
    def num_projections(self) -> int:
        """
        Get the number of linear maps of the input: the gates and the candidate state.

        Returns
        -------
        int
            The number of maps.
        """
        match self:
            case MinimalRecurrentCell.GRU:
                return 2
            case MinimalRecurrentCell.LSTM:
                return 3
