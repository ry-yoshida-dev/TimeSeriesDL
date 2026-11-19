type EncodedValue = None | bool | int | float | str | list[EncodedValue] | dict[str, EncodedValue]
"""A value that ``torch.load`` reads back with ``weights_only=True``, so loading it cannot run code."""
