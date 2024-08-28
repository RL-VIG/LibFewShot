import torch
from torch import nn
import numpy as np

from .finetuning_model import FinetuningModel


class IfslPretrain(FinetuningModel):
    def __init__(self,feat_dim, num_class, inner_param,ifsl_pretrain_param,
                 emb_func_path = None,cls_classifier_path = None, **kwargs):
        super(IfslPretrain, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.inner_param = inner_param
        for (key,value) in ifsl_pretrain_param.items():
            setattr(self, key, value)
        self.emb_func =self._load_state_dict(self.emb_func,emb_func_path)
        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.classifier=self._load_state_dict(self.classifier,cls_classifier_path)
        self.loss_func = nn.CrossEntropyLoss()
        self.features = np.zeros((self.num_class, self.feat_dim))
        self.temp_features = np.zeros((self.num_class, self.feat_dim))
        self.counts = np.zeros((self.num_class))


    def _load_state_dict(self, model, state_dict_path):
        if state_dict_path is not None:
            model_state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model_state_dict)
        return model

    def set_forward(self, batch):
        """
        :param batch:
        :return:
        """

        return 0,0

    def normalize(self, x, dim=1):
        x_norm = torch.norm(x, p=2, dim=dim).unsqueeze(dim).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        feat = self.emb_func(image)
        output = self.classifier(feat)
        if self.norm is True:
            feat=self.normalize(feat)
        if self.featuring is True:
            fs = feat.detach().cpu().numpy()
            ids=target.detach().cpu().numpy()
            for i in range(ids.shape[0]):
                self.temp_features[ids[i]]+=fs[i]
                self.counts[ids[i]]+=1
            for i in range(self.num_class):
                if self.counts[i]>0:
                    self.features[i]=self.temp_features[i]/self.counts[i]
            np.save(self.feature_path,self.features)
        loss =(0 if self.featuring is True else 1.0) * self.loss_func(output, target)
        acc = 100 * torch.count_nonzero(torch.argmax(output, dim=1) == target).detach().cpu().item() / target.size(0)
        return output, acc, loss

    def set_forward_adaptation(self):
        pass
