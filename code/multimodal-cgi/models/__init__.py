from .MultiModal_CGI import Multimodal_CGI
from .disp2prob import *

from .loss import model_loss_train, model_loss_test, model_loss_train_v2, model_loss_train_v3


__models__ = {
    "Multimodal_CGI": Multimodal_CGI
}
