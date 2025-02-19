from .selfblended import Detector as SelfBlendedModel
from .multi_attention import  Detector as MAT
from .ict import Detector as ICT
from .ict_utils import calculate_roc_ex, evaluate_new
from .rossler import  TransferModel as RosslerModel
from .forgerynet import  xception as ForgeryNet
from .caddm import CADDM
from .resnet import resnet50