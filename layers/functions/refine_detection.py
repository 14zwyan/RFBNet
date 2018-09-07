import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from utils.box_utils import decode


class RefineDetect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, cfg,obj_thresh=0.01):
        self.num_classes = num_classes
        self.background_label = bkg_label

        self.variance = cfg['variance']
        self.obj_thresh = obj_thresh

    def forward(self, predictions, prior):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        loc, conf,obj  = predictions

        loc_data = loc.data
        conf_data = conf.data
        obj_data = obj.data
        prior_data = prior.data

        no_obj_index = obj_data[:,:,1] < self.obj_thresh
        #print(conf_data.shape)
        #print(no_obj_index.shape)
        conf_data[ no_obj_index.unsqueeze(2).expand_as(conf_data)] = 0

        num = loc_data.size(0)  # batch size
        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(1, self.num_priors, 4)
        self.scores = torch.zeros(1, self.num_priors, self.num_classes)
        self.obj = torch.zeros(1,self.num_priors,2)
        if loc_data.is_cuda:
            self.boxes = self.boxes.cuda()
            self.scores = self.scores.cuda()
            self.obj =self.obj.cuda()

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)
            obj_preds = obj_data.unsqueeze(0)

        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes)
            obj_preds = obj_data.view(nu,num_priors,2)
            self.boxes.expand_(num, self.num_priors, 4)
            self.scores.expand_(num, self.num_priors, self.num_classes)
            self.obj.expand_(num,self.num_priors,2)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            obj_scores = obj_preds[i].clone()

            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores
            self.obj[i] = obj_scores

        return self.boxes, self.scores,self.obj
