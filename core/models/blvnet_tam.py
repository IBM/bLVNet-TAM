import torch.nn as nn
import torch

from .blvnet_tam_backbone import blvnet_tam_backbone
from ._model_urls import model_urls


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class bLVNet_TAM(nn.Module):

    def __init__(self, params):
        super().__init__()
        params = DotDict(params)
        print(params)
        self.baseline_model = blvnet_tam_backbone(params.depth, params.alpha, params.beta,
                                                  num_frames=params.groups,
                                                  blending_frames=params.blending_frames,
                                                  input_channels=params.input_channels,
                                                  imagenet_blnet_pretrained=params.imagenet_blnet_pretrained)
        self.num_frames = params.groups
        self.blending_frames = params.blending_frames
        self.blending_method = params.blending_method
        self.partial_freeze_bn = params.partial_freeze_bn
        self.dropout = params.dropout
        self.modality = 'rgb'

        # new category
        self.consensus = SegmentConsensus(consensus_type='avg', dim=1)
        # get the dim of feature vec
        feature_dim = getattr(self.baseline_model, 'fc').in_features
        # update the fc layer and initialize it
        self.prepare_baseline(feature_dim, params.num_classes)

        self.model_name = '{dataset}-bLVNet-TAM-{depth}-a{alpha}-b{beta}-f{num_frames}x2'.format(
            dataset=params.dataset, depth=params.depth, alpha=params.alpha, beta=params.beta,
            num_frames=params.groups // 2)

        if params.pretrained:
            checkpoint = torch.load(model_urls[self.model_name], map_location='cpu')
            self.load_state_dict(checkpoint)

    def prepare_baseline(self, feature_dim, num_classes):
        if self.dropout > 0.0:
            # replace the original fc layer as dropout layer
            setattr(self.baseline_model, 'fc', nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_classes)
            nn.init.normal_(self.new_fc.weight, 0, 0.001)
            nn.init.constant_(self.new_fc.bias, 0)
        else:
            setattr(self.baseline_model, 'fc', nn.Linear(feature_dim, num_classes))
            nn.init.normal_(getattr(self.baseline_model, 'fc').weight, 0, 0.001)
            nn.init.constant_(getattr(self.baseline_model, 'fc').bias, 0)

    def forward(self, x):
        n, c_t, h, w = x.shape
        batched_input = x.view(n * self.num_frames, c_t // self.num_frames, h, w)
        base_out = self.baseline_model(batched_input)
        if self.dropout > 0.0:
            base_out = self.new_fc(base_out)
        n_t, c = base_out.shape
        curr_num_frames = n_t // n
        base_out = base_out.view(n, curr_num_frames, c)
        # average all frames
        out = self.consensus(base_out)
        # dim of out: [N, 1, num_classes]
        out = out.squeeze(1)
        # dim of out: [N, num_classes]
        return out


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in
