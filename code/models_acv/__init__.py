from models_acv.CGF_ACV import ACVNet
from models_acv.loss import acv_model_loss_train, acv_model_loss_test, acv_model_loss_train_attn_only, acv_model_loss_train_freeze_attn

__tmodels__ = {
    "acvnet": ACVNet
}
