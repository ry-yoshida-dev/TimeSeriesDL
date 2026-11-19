# irregular_heads

## Overview

Generic output layers around any `IrregularSequenceEncoder`, so an encoder-based continuous-time model only defines
its encoder.

## Components

| Component | Description |
| --------- | ----------- |
| [head.py](./head.py) | `IrregularEncoderHead`: hidden sizes, activation and dropout of the output layers, and their builder. |
| [irregular_summary_network.py](./irregular_summary_network.py) | `EncodedIrregularSummaryNetwork`: encoder features mapped to the output vector. |
