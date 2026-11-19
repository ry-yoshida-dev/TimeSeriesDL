"""Model parameters for rebuilding a network whose weights a checkpoint restores."""

from ..models.interfaces.protocols.published_weights import PublishedWeightsModel


def restorable_model[ModelT](model: ModelT) -> ModelT:
    """
    Get the parameters that rebuild a saved network before its weights are restored from a checkpoint.

    Published weights are dropped, since the checkpoint's weights replace them: a saved task then loads without
    the Hugging Face Hub or its cache. Other parameters are returned unchanged.

    Parameters
    ----------
    model : ModelT
        The saved model parameters.

    Returns
    -------
    ModelT
        The same architecture, building randomly initialized networks.
    """
    if isinstance(model, PublishedWeightsModel):
        return model.without_pretrained_weights()
    return model
