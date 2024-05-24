from .CGI_Stereo import CGI_Stereo
from .loss import model_loss_train, model_loss_test,KD_feat_loss,KD_cvolume_loss,KD_deconv4,KD_deconv8
# from .autoencoder import dec
# from .mgd import MGDLoss

__t_models__ = {
    "CGI_Stereo": CGI_Stereo   
}
