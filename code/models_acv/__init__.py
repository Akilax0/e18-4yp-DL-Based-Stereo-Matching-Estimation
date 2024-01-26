from models_acv.acv import ACVNet
from models_acv.loss import acv_model_loss_train_attn_only, acv_model_loss_train_freeze_attn, acv_model_loss_train, acv_model_loss_test

__t_models__ = {
    "acvnet": ACVNet
}
