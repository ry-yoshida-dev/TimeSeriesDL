from enum import StrEnum

from ..architectures.continuous_time.latent_ode import LatentODEParameters
from ..architectures.continuous_time.mtan import MTANParameters
from ..architectures.continuous_time.neural_cde import NeuralCDEParameters
from ..architectures.continuous_time.ode_rnn import ODERNNParameters
from ..architectures.convolutional.fcn import FCNParameters
from ..architectures.convolutional.inception_time import InceptionTimeParameters
from ..architectures.convolutional.lite import LITEParameters
from ..architectures.convolutional.micn import MICNParameters
from ..architectures.convolutional.mini_rocket import MiniRocketParameters
from ..architectures.convolutional.modern_tcn import ModernTCNParameters
from ..architectures.convolutional.multi_rocket import MultiRocketParameters
from ..architectures.convolutional.resnet import ResNetParameters
from ..architectures.convolutional.rocket import RocketParameters
from ..architectures.convolutional.tcn import TCNParameters
from ..architectures.convolutional.times_net import TimesNetParameters
from ..architectures.convolutional.ts2vec import TS2VecParameters
from ..architectures.convolutional.ts_tcc import TSTCCParameters
from ..architectures.foundation.chronos import ChronosParameters
from ..architectures.foundation.chronos_bolt import ChronosBoltParameters
from ..architectures.foundation.moirai import MoiraiParameters
from ..architectures.foundation.moment import MOMENTParameters
from ..architectures.foundation.timer import TimerParameters
from ..architectures.foundation.times_fm import TimesFMParameters
from ..architectures.foundation.ttm import TTMParameters
from ..architectures.graph.agcrn import AGCRNParameters
from ..architectures.graph.fourier_gnn import FourierGNNParameters
from ..architectures.graph.graph_wavenet import GraphWaveNetParameters
from ..architectures.graph.mtgnn import MTGNNParameters
from ..architectures.graph.stem_gnn import StemGNNParameters
from ..architectures.linear.cycle_net import CycleNetParameters
from ..architectures.linear.dlinear import DLinearParameters
from ..architectures.linear.fits import FITSParameters
from ..architectures.linear.nlinear import NLinearParameters
from ..architectures.linear.rlinear import RLinearParameters
from ..architectures.linear.sparse_tsf import SparseTSFParameters
from ..architectures.mlp.frets import FreTSParameters
from ..architectures.mlp.koopa import KoopaParameters
from ..architectures.mlp.n_beats import NBEATSParameters
from ..architectures.mlp.n_hits import NHITSParameters
from ..architectures.mlp.tide import TiDEParameters
from ..architectures.mlp.time_mixer import TimeMixerParameters
from ..architectures.mlp.time_mixer_plus_plus import TimeMixerPlusPlusParameters
from ..architectures.mlp.ts_mixer import TSMixerParameters
from ..architectures.mlp.usad import USADParameters
from ..architectures.mlp.vanilla import MLPParameters
from ..architectures.pretrained.encoder import PretrainedEncoderParameters
from ..architectures.recurrent.brits import BRITSParameters
from ..architectures.recurrent.deep_ar import DeepARParameters
from ..architectures.recurrent.gru_d import GRUDParameters
from ..architectures.recurrent.lstnet import LSTNetParameters
from ..architectures.recurrent.min_rnn import MinRNNParameters
from ..architectures.recurrent.seg_rnn import SegRNNParameters
from ..architectures.recurrent.time_gan import TimeGANParameters
from ..architectures.recurrent.time_grad import TimeGradParameters
from ..architectures.recurrent.vanilla import RNNParameters
from ..architectures.recurrent.xlstm import XLSTMParameters
from ..architectures.recurrent.xlstm_time import XLSTMTimeParameters
from ..architectures.state_space.film import FiLMParameters
from ..architectures.state_space.hippo_rnn import HiPPORNNParameters
from ..architectures.state_space.lmu import LMUParameters
from ..architectures.state_space.lru import LRUParameters
from ..architectures.state_space.mamba import MambaParameters
from ..architectures.state_space.mamba2 import Mamba2Parameters
from ..architectures.state_space.s4 import S4Parameters
from ..architectures.state_space.s4d import S4DParameters
from ..architectures.state_space.s5 import S5Parameters
from ..architectures.state_space.s_mamba import SMambaParameters
from ..architectures.transformer.anomaly_transformer import AnomalyTransformerParameters
from ..architectures.transformer.autoformer import AutoformerParameters
from ..architectures.transformer.conv_tran import ConvTranParameters
from ..architectures.transformer.crossformer import CrossformerParameters
from ..architectures.transformer.csdi import CSDIParameters
from ..architectures.transformer.dcdetector import DCdetectorParameters
from ..architectures.transformer.diffusion_ts import DiffusionTSParameters
from ..architectures.transformer.etsformer import ETSformerParameters
from ..architectures.transformer.fedformer import FEDformerParameters
from ..architectures.transformer.informer import InformerParameters
from ..architectures.transformer.itransformer import ITransformerParameters
from ..architectures.transformer.non_stationary_transformer import NonStationaryTransformerParameters
from ..architectures.transformer.patch_tst import PatchTSTParameters
from ..architectures.transformer.pyraformer import PyraformerParameters
from ..architectures.transformer.saits import SAITSParameters
from ..architectures.transformer.temporal_fusion_transformer import TemporalFusionTransformerParameters
from ..architectures.transformer.time_xer import TimeXerParameters
from ..architectures.transformer.tranad import TranADParameters
from ..architectures.transformer.tst import TSTParameters
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
    | HiPPORNNParameters
    | LMUParameters
    | S4Parameters
    | S4DParameters
    | S5Parameters
    | LRUParameters
    | MambaParameters
    | Mamba2Parameters
    | SMambaParameters
    | FiLMParameters
    | RLinearParameters
    | FITSParameters
    | SparseTSFParameters
    | TimeMixerParameters
    | TimeMixerPlusPlusParameters
    | FreTSParameters
    | KoopaParameters
    | XLSTMParameters
    | XLSTMTimeParameters
    | MinRNNParameters
    | LSTNetParameters
    | MICNParameters
    | ModernTCNParameters
    | LITEParameters
    | RocketParameters
    | MiniRocketParameters
    | MultiRocketParameters
    | NonStationaryTransformerParameters
    | PyraformerParameters
    | ETSformerParameters
    | CrossformerParameters
    | TimeXerParameters
    | TSTParameters
    | ConvTranParameters
    | MTGNNParameters
    | GraphWaveNetParameters
    | AGCRNParameters
    | StemGNNParameters
    | FourierGNNParameters
    | SAITSParameters
    | BRITSParameters
    | GRUDParameters
    | TS2VecParameters
    | TSTCCParameters
    | PretrainedEncoderParameters
    | USADParameters
    | AnomalyTransformerParameters
    | DCdetectorParameters
    | TranADParameters
    | MTANParameters
    | NeuralCDEParameters
    | ODERNNParameters
    | LatentODEParameters
    | TimeGradParameters
    | CSDIParameters
    | ChronosBoltParameters
    | ChronosParameters
    | TimesFMParameters
    | MoiraiParameters
    | MOMENTParameters
    | TimerParameters
    | TTMParameters
    | CycleNetParameters
    | TimeGANParameters
    | DiffusionTSParameters
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
    HIPPO_RNN = "hippo_rnn"
    LMU = "lmu"
    S4 = "s4"
    S4D = "s4d"
    S5 = "s5"
    LRU = "lru"
    MAMBA = "mamba"
    MAMBA2 = "mamba2"
    S_MAMBA = "s_mamba"
    FILM = "film"
    RLINEAR = "rlinear"
    FITS = "fits"
    SPARSE_TSF = "sparse_tsf"
    TIME_MIXER = "time_mixer"
    TIME_MIXER_PLUS_PLUS = "time_mixer_plus_plus"
    FRETS = "frets"
    KOOPA = "koopa"
    XLSTM = "xlstm"
    XLSTM_TIME = "xlstm_time"
    MIN_RNN = "min_rnn"
    LSTNET = "lstnet"
    MICN = "micn"
    MODERN_TCN = "modern_tcn"
    LITE = "lite"
    ROCKET = "rocket"
    MINI_ROCKET = "mini_rocket"
    MULTI_ROCKET = "multi_rocket"
    NON_STATIONARY_TRANSFORMER = "non_stationary_transformer"
    PYRAFORMER = "pyraformer"
    ETSFORMER = "etsformer"
    CROSSFORMER = "crossformer"
    TIME_XER = "time_xer"
    TST = "tst"
    CONV_TRAN = "conv_tran"
    MTGNN = "mtgnn"
    GRAPH_WAVENET = "graph_wavenet"
    AGCRN = "agcrn"
    STEM_GNN = "stem_gnn"
    FOURIER_GNN = "fourier_gnn"
    SAITS = "saits"
    BRITS = "brits"
    GRU_D = "gru_d"
    TS2VEC = "ts2vec"
    TS_TCC = "ts_tcc"
    PRETRAINED_ENCODER = "pretrained_encoder"
    USAD = "usad"
    ANOMALY_TRANSFORMER = "anomaly_transformer"
    DCDETECTOR = "dcdetector"
    TRANAD = "tranad"
    MTAN = "mtan"
    NEURAL_CDE = "neural_cde"
    ODE_RNN = "ode_rnn"
    LATENT_ODE = "latent_ode"
    TIME_GRAD = "time_grad"
    CSDI = "csdi"
    CHRONOS_BOLT = "chronos_bolt"
    CHRONOS = "chronos"
    TIMES_FM = "times_fm"
    MOIRAI = "moirai"
    MOMENT = "moment"
    TIMER = "timer"
    TTM = "ttm"
    CYCLE_NET = "cycle_net"
    TIME_GAN = "time_gan"
    DIFFUSION_TS = "diffusion_ts"

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
            case ModelArchitecture.HIPPO_RNN:
                return HiPPORNNParameters
            case ModelArchitecture.LMU:
                return LMUParameters
            case ModelArchitecture.S4:
                return S4Parameters
            case ModelArchitecture.S4D:
                return S4DParameters
            case ModelArchitecture.S5:
                return S5Parameters
            case ModelArchitecture.LRU:
                return LRUParameters
            case ModelArchitecture.MAMBA:
                return MambaParameters
            case ModelArchitecture.MAMBA2:
                return Mamba2Parameters
            case ModelArchitecture.S_MAMBA:
                return SMambaParameters
            case ModelArchitecture.FILM:
                return FiLMParameters
            case ModelArchitecture.RLINEAR:
                return RLinearParameters
            case ModelArchitecture.FITS:
                return FITSParameters
            case ModelArchitecture.SPARSE_TSF:
                return SparseTSFParameters
            case ModelArchitecture.TIME_MIXER:
                return TimeMixerParameters
            case ModelArchitecture.TIME_MIXER_PLUS_PLUS:
                return TimeMixerPlusPlusParameters
            case ModelArchitecture.FRETS:
                return FreTSParameters
            case ModelArchitecture.KOOPA:
                return KoopaParameters
            case ModelArchitecture.XLSTM:
                return XLSTMParameters
            case ModelArchitecture.XLSTM_TIME:
                return XLSTMTimeParameters
            case ModelArchitecture.MIN_RNN:
                return MinRNNParameters
            case ModelArchitecture.LSTNET:
                return LSTNetParameters
            case ModelArchitecture.MICN:
                return MICNParameters
            case ModelArchitecture.MODERN_TCN:
                return ModernTCNParameters
            case ModelArchitecture.LITE:
                return LITEParameters
            case ModelArchitecture.ROCKET:
                return RocketParameters
            case ModelArchitecture.MINI_ROCKET:
                return MiniRocketParameters
            case ModelArchitecture.MULTI_ROCKET:
                return MultiRocketParameters
            case ModelArchitecture.NON_STATIONARY_TRANSFORMER:
                return NonStationaryTransformerParameters
            case ModelArchitecture.PYRAFORMER:
                return PyraformerParameters
            case ModelArchitecture.ETSFORMER:
                return ETSformerParameters
            case ModelArchitecture.CROSSFORMER:
                return CrossformerParameters
            case ModelArchitecture.TIME_XER:
                return TimeXerParameters
            case ModelArchitecture.TST:
                return TSTParameters
            case ModelArchitecture.CONV_TRAN:
                return ConvTranParameters
            case ModelArchitecture.MTGNN:
                return MTGNNParameters
            case ModelArchitecture.GRAPH_WAVENET:
                return GraphWaveNetParameters
            case ModelArchitecture.AGCRN:
                return AGCRNParameters
            case ModelArchitecture.STEM_GNN:
                return StemGNNParameters
            case ModelArchitecture.FOURIER_GNN:
                return FourierGNNParameters
            case ModelArchitecture.SAITS:
                return SAITSParameters
            case ModelArchitecture.BRITS:
                return BRITSParameters
            case ModelArchitecture.GRU_D:
                return GRUDParameters
            case ModelArchitecture.TS2VEC:
                return TS2VecParameters
            case ModelArchitecture.TS_TCC:
                return TSTCCParameters
            case ModelArchitecture.PRETRAINED_ENCODER:
                return PretrainedEncoderParameters
            case ModelArchitecture.USAD:
                return USADParameters
            case ModelArchitecture.ANOMALY_TRANSFORMER:
                return AnomalyTransformerParameters
            case ModelArchitecture.DCDETECTOR:
                return DCdetectorParameters
            case ModelArchitecture.TRANAD:
                return TranADParameters
            case ModelArchitecture.MTAN:
                return MTANParameters
            case ModelArchitecture.NEURAL_CDE:
                return NeuralCDEParameters
            case ModelArchitecture.ODE_RNN:
                return ODERNNParameters
            case ModelArchitecture.LATENT_ODE:
                return LatentODEParameters
            case ModelArchitecture.TIME_GRAD:
                return TimeGradParameters
            case ModelArchitecture.CSDI:
                return CSDIParameters
            case ModelArchitecture.CHRONOS_BOLT:
                return ChronosBoltParameters
            case ModelArchitecture.CHRONOS:
                return ChronosParameters
            case ModelArchitecture.TIMES_FM:
                return TimesFMParameters
            case ModelArchitecture.MOIRAI:
                return MoiraiParameters
            case ModelArchitecture.MOMENT:
                return MOMENTParameters
            case ModelArchitecture.TIMER:
                return TimerParameters
            case ModelArchitecture.TTM:
                return TTMParameters
            case ModelArchitecture.CYCLE_NET:
                return CycleNetParameters
            case ModelArchitecture.TIME_GAN:
                return TimeGANParameters
            case ModelArchitecture.DIFFUSION_TS:
                return DiffusionTSParameters
