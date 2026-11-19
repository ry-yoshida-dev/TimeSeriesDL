from enum import StrEnum

from ..architectures.convolutional.fcn import FCNParameters
from ..architectures.convolutional.inception_time import InceptionTimeParameters
from ..architectures.convolutional.resnet import ResNetParameters
from ..architectures.convolutional.tcn import TCNParameters
from ..architectures.convolutional.times_net import TimesNetParameters
from ..architectures.linear.dlinear import DLinearParameters
from ..architectures.linear.nlinear import NLinearParameters
from ..architectures.mlp.n_beats import NBEATSParameters
from ..architectures.mlp.n_hits import NHITSParameters
from ..architectures.mlp.tide import TiDEParameters
from ..architectures.mlp.ts_mixer import TSMixerParameters
from ..architectures.mlp.vanilla import MLPParameters
from ..architectures.recurrent.deep_ar import DeepARParameters
from ..architectures.recurrent.seg_rnn import SegRNNParameters
from ..architectures.recurrent.vanilla import RNNParameters
from ..architectures.transformer.autoformer import AutoformerParameters
from ..architectures.transformer.fedformer import FEDformerParameters
from ..architectures.transformer.informer import InformerParameters
from ..architectures.transformer.itransformer import ITransformerParameters
from ..architectures.transformer.patch_tst import PatchTSTParameters
from ..architectures.transformer.temporal_fusion_transformer import TemporalFusionTransformerParameters
from ..architectures.transformer.vanilla import TransformerParameters

type ModelParameters = (
    MLPParameters
    | RNNParameters
    | TCNParameters
    | FCNParameters
    | ResNetParameters
    | InceptionTimeParameters
    | DLinearParameters
    | NLinearParameters
    | TSMixerParameters
    | TiDEParameters
    | SegRNNParameters
    | TransformerParameters
    | InformerParameters
    | AutoformerParameters
    | FEDformerParameters
    | PatchTSTParameters
    | ITransformerParameters
    | TimesNetParameters
    | NBEATSParameters
    | NHITSParameters
    | DeepARParameters
    | TemporalFusionTransformerParameters
)


class ModelArchitecture(StrEnum):
    """
    Model family; each value names the parameters class it is configured with.

    Values are the names of the model sub-packages and of the YAML presets in ``config/``.
    """

    MLP = "mlp"
    RNN = "rnn"
    TCN = "tcn"
    FCN = "fcn"
    RESNET = "resnet"
    INCEPTION_TIME = "inception_time"
    DLINEAR = "dlinear"
    NLINEAR = "nlinear"
    TS_MIXER = "ts_mixer"
    TIDE = "tide"
    SEG_RNN = "seg_rnn"
    TRANSFORMER = "transformer"
    INFORMER = "informer"
    AUTOFORMER = "autoformer"
    FEDFORMER = "fedformer"
    PATCH_TST = "patch_tst"
    ITRANSFORMER = "itransformer"
    TIMES_NET = "times_net"
    N_BEATS = "n_beats"
    N_HITS = "n_hits"
    DEEP_AR = "deep_ar"
    TEMPORAL_FUSION_TRANSFORMER = "temporal_fusion_transformer"

    @property
    def parameters_type(self) -> type[ModelParameters]:
        """
        Parameters class of the architecture.

        Returns
        -------
        type[ModelParameters]
            Dataclass whose fields configure the architecture, such as ``PatchTSTParameters``.
        """
        match self:
            case ModelArchitecture.MLP:
                return MLPParameters
            case ModelArchitecture.RNN:
                return RNNParameters
            case ModelArchitecture.TCN:
                return TCNParameters
            case ModelArchitecture.FCN:
                return FCNParameters
            case ModelArchitecture.RESNET:
                return ResNetParameters
            case ModelArchitecture.INCEPTION_TIME:
                return InceptionTimeParameters
            case ModelArchitecture.DLINEAR:
                return DLinearParameters
            case ModelArchitecture.NLINEAR:
                return NLinearParameters
            case ModelArchitecture.TS_MIXER:
                return TSMixerParameters
            case ModelArchitecture.TIDE:
                return TiDEParameters
            case ModelArchitecture.SEG_RNN:
                return SegRNNParameters
            case ModelArchitecture.TRANSFORMER:
                return TransformerParameters
            case ModelArchitecture.INFORMER:
                return InformerParameters
            case ModelArchitecture.AUTOFORMER:
                return AutoformerParameters
            case ModelArchitecture.FEDFORMER:
                return FEDformerParameters
            case ModelArchitecture.PATCH_TST:
                return PatchTSTParameters
            case ModelArchitecture.ITRANSFORMER:
                return ITransformerParameters
            case ModelArchitecture.TIMES_NET:
                return TimesNetParameters
            case ModelArchitecture.N_BEATS:
                return NBEATSParameters
            case ModelArchitecture.N_HITS:
                return NHITSParameters
            case ModelArchitecture.DEEP_AR:
                return DeepARParameters
            case ModelArchitecture.TEMPORAL_FUSION_TRANSFORMER:
                return TemporalFusionTransformerParameters
