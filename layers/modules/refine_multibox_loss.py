import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
GPU = False
if torch.cuda.is_available():
    GPU = True


class RefineMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """


    def __init__(self, num_classes,overlap_thresh,prior_for_matching,bkg_label,neg_mining,neg_pos,neg_overlap,encode_target,obj_score):
        super(RefineMultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1,0.2]
        self.obj_score = obj_score

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        #[ batch_size,num_priors,4 ]
        #[batch_size,num_priors,21]
        #[batch_size,num_priors,2]
        loc_data, conf_data, obj_data = predictions
        priors = priors
        #bath_size 
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        obj_t = torch.LongTensor(num,num_priors)
        for idx in range(num):
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            defaults = priors.data
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)
            obj_t = conf_t.clone() 
            obj_t [  conf_t>0 ] = 1
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            obj_t = obj_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)
        obj_t = Variable(obj_t,requires_grad=False)



        obj_conf_data = obj_data[:,:,1].detach()
        #print(obj_data.requires_grad)
        #print(obj_conf_data.requires_grad)
        pos = conf_t > 0
        #print(type(pos))
        neg_positive = obj_conf_data < self.obj_score
        #print(type(neg_positive))
        neg_positive = (pos + neg_positive) > 2 
        pos = pos - neg_positive
        #print(pos.type())
        #byte tensor  
        # for pose conf_t  > 0 ---> 1 
        # for neg_positive  conf < obj_score --> 1 
        # 1 1 -> 0  # focus 
        # 1 0 -> 1   
        # 0 1 -> 0 
        # 0 0 -> 0 
        #print(type(pos))

        #pos = ( conf_t > 0 ) - (obj_conf_data <= self.obj_score)
        #pos[ (obj_conf_data < self.obj_score).detach()] = 0 

        if pos.data.long().sum() == 0:
            pos = conf_t > 0

        #print('conf_t shape:'+str(conf_t.shape))
        #print('conf_t >0 shape:'+str( (conf_t>0).sum()))
        #print('obj_t > obj_score'+str( (obj_t > self.obj_score).sum() ))
        #print('pos shape:'+str( pos.sum() ))

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        pos_obj = obj_t > 0
        #print('pos_obj shape: '+str(pos_obj.shape))
        batch_obj_conf = obj_data.view(-1,2)
        loss_obj = log_sum_exp(batch_obj_conf) - batch_obj_conf.gather(1, obj_t.view(-1,1))
        
        loss_obj[ pos_obj] = 0 
        loss_obj = loss_obj.view(num,-1)
        _, loss_obj_idx = loss_obj.sort(1,descending=True)
        _, idx_obj_rank = loss_obj_idx.sort(1)
        num_obj_pos = pos_obj.long().sum(1,keepdim=True)
        num_obj_neg = torch.clamp( self.negpos_ratio*num_obj_pos,max=pos_obj.size(1)-1)
        #print('num_obj_pos:'+str(num_obj_pos.shape))
        #print('num_obj_neg:'+str(num_obj_neg.shape))
        neg_obj = idx_obj_rank < num_obj_neg.expand_as(idx_obj_rank)

        pos_obj_idx = pos_obj.unsqueeze(2).expand_as(obj_data)
        neg_obj_idx = neg_obj.unsqueeze(2).expand_as(obj_data)

        conf_obj_p = obj_data[ (pos_obj_idx+neg_obj_idx).gt(0) ].view(-1,2)
        targets_weighted = obj_t[(pos_obj+neg_obj).gt(0)]
        loss_obj = F.cross_entropy( conf_obj_p,targets_weighted,size_average=False )

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1,self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))

        # Hard Negative Mining
        loss_c[pos.view(-1,1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _,loss_idx = loss_c.sort(1, descending=True)
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        #print('num_pos:'+str(num_pos.sum()))
        #print('num_neg:'+str(num_neg.sum()))
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        #print('conf_p.shape'+str(conf_p.shape))
        #print('targets_weighted'+str(targets_weighted.shape))
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = max(num_pos.data.sum(), 1)
        loss_l/=N*1.0
        loss_c/=N*1.0
        N1 = max(num_obj_neg.data.sum(),1)
        loss_obj /= N1
        loss_obj = 0.4 * loss_obj

        return loss_l,loss_c,loss_obj
