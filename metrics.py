import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor

class DiceCeLoss(nn.Module):
    # predict : output of model (i.e. no softmax)[N,C,*]
    # target : gt of img [N,1,*]
    def __init__(self, num_classes, alpha=1.0,weighted=True):
        '''
        calculate loss:
            celoss + alpha*celoss
            alpha : default is 1
        '''
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = Diceloss(self.num_classes)
        if weighted:
            self.celoss = WeightedCrossEntropyLoss(self.num_classes)
        else:
            self.celoss = RobustCrossEntropyLoss()

    def forward(self, predict, label):
        # predict is output of the model, i.e. without softmax [N,C,*]
        # label is not one hot encoding [N,1,*]

        diceloss = self.diceloss(predict, label)
        celoss = self.celoss(predict, label)
        loss = celoss + self.alpha * diceloss
        return diceloss,celoss,loss



class Diceloss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def one_hot_encode(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor == i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, input, target, weight=None, softmax=True):
        if softmax:
            inputs = F.softmax(input, dim=1)
        target = self.one_hot_encode(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.shape == target.shape, 'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(diceloss)
            loss += diceloss * weight[i]
        loss = loss / self.n_classes
        return loss





class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eps = 1e-4
        self.num_classes = num_classes

    def forward(self, predict, target):
        weight = []
        for c in range(self.num_classes):
            weight_c = torch.sum(target == c).float()
            #print("weightc for c",c,weight_c)
            weight.append(weight_c)
        weight = torch.tensor(weight).to(target.device)
        weight = 1 - weight / (torch.sum(weight))
        if len(target.shape) == len(predict.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        wce_loss = F.cross_entropy(predict, target.long(), weight)
        return wce_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def prototype_loss(feature,feature_t,label,num_cls):
    """
    match prototype simarlity map for teacher and student model
    feature/feature_t [N,C,*]
    label [N,1,*]
    """
    eps = 1e-5
    N = len(feature.size()) - 2
    label = label[:,0]  #label [N,*]
    s = []
    t = []


    for i in range(num_cls):
        mask = label==i
        if N==3 and (torch.sum(mask,dim=(-3,-2,-1))>0).all():
            proto_s =  torch.sum(feature*mask[:,None],dim=(-3,-2,-1))/(torch.sum(mask[:,None],dim=(-3,-2,-1))+eps)
            proto_t =  torch.sum(feature_t*mask[:,None],dim=(-3,-2,-1))/(torch.sum(mask[:,None],dim=(-3,-2,-1))+eps)
            proto_map_s = F.cosine_similarity(feature,proto_s[:,:,None,None,None],dim=1,eps=eps)
            proto_map_t = F.cosine_similarity(feature_t,proto_t[:,:,None,None,None],dim=1,eps=eps)
            s.append(proto_map_s.unsqueeze(1))
            t.append(proto_map_t.unsqueeze(1))
    sim_map_s = torch.cat(s,dim=1)
    sim_map_t = torch.cat(t,dim=1)
    loss = torch.mean((sim_map_s-sim_map_t)**2)
    return sim_map_s,sim_map_t,loss


def vae_loss(recon_x, x, mu, logvar):
    loss_dict = {}
    loss_dict['KLD'] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss_dict['recon_loss'] = F.mse_loss(recon_x, x, reduction='mean')

    return loss_dict




class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def D_basic(netD, real, fake, criterionGAN):
    # Real
    pred_real = netD(real)
    loss_D_real = criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake = criterionGAN(pred_fake, False)
    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    # backward
    # loss_D.backward()
    return loss_D

def central_moment_discrepancy(feature1, feature2):
    # 计算均值
    mean1 = torch.mean(feature1, dim=(2, 3, 4), keepdim=True)
    mean2 = torch.mean(feature2, dim=(2, 3, 4), keepdim=True)

    # 计算方差
    var1 = torch.mean((feature1 - mean1) ** 2, dim=(2, 3, 4), keepdim=True)
    var2 = torch.mean((feature2 - mean2) ** 2, dim=(2, 3, 4), keepdim=True)

    # 计算偏度
    skew1 = torch.mean((feature1 - mean1) ** 3, dim=(2, 3, 4), keepdim=True)
    skew2 = torch.mean((feature2 - mean2) ** 3, dim=(2, 3, 4), keepdim=True)

    # 计算峰度
    kurtosis1 = torch.mean((feature1 - mean1) ** 4, dim=(2, 3, 4), keepdim=True)
    kurtosis2 = torch.mean((feature2 - mean2) ** 4, dim=(2, 3, 4), keepdim=True)

    cmd_loss = torch.mean(
        (mean1 - mean2) ** 2 + (var1 - var2) ** 2 + (skew1 - skew2) ** 2 + (kurtosis1 - kurtosis2) ** 2)

    return cmd_loss











def dice(im1, im2,tid):
    im1=im1==tid
    im2=im2==tid
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc

def dice_loss(target, predictive, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss
# class DiceLoss(nn.Module):
#     def __init__(self, n_classes):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes
#
#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob)
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()
#
#     def _dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss
#
#     def forward(self, inputs, target, weight=None, softmax=False):
#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         target = self._one_hot_encoder(target)
#         if weight is None:
#             weight = [1] * self.n_classes
#         assert inputs.size() == target.size(), 'predict & target shape do not match'
#         class_wise_dice = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             dice = self._dice_loss(inputs[:, i], target[:, i])
#             class_wise_dice.append(1.0 - dice.item())
#             loss += dice * weight[i]
#         return loss / self.n_classes

def cc1(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1).float()
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1).float()

        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target.pow(self.p)) * valid_mask, dim=1) + self.smooth
        # a = 0.5
        # b = 0.7
        # num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) + self.smooth
        # den = torch.sum(torch.mul(predict, target) + a*torch.mul(predict, (1-target))) + b*torch.mul((1-predict), target)*valid_mask + self.smooth


        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))










class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input"""

    def __init__(self, weight=None, aux=False, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight

    def _base_forward(self, predict, target, valid_mask):

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[-1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[..., i], valid_mask)
                if self.weight is not None:
                    # print(target.shape)
                    assert self.weight.shape[0] == target.shape[-1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[-1], self.weight.shape[0])
                    # print(dice_loss, self.weight[i])
                    dice_loss *= self.weight[i]
                    # print(dice_loss)
                total_loss += dice_loss

        return total_loss / target.shape[-1]

    def _aux_forward(self, output, target, **kwargs):
        # *preds, target = tuple(inputs)
        valid_mask = (target != self.ignore_index).long()
        target_one_hot = F.one_hot(torch.clamp_min(target, 0))
        loss = self._base_forward(output[0], target_one_hot, valid_mask)
        for i in range(1, len(output)):
            aux_loss = self._base_forward(output[i], target_one_hot, valid_mask)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, output, target):
        # preds, target = tuple(inputs)
        # inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(output, target)
        else:
            valid_mask = (target != self.ignore_index).long()
            target_one_hot = F.one_hot(torch.clamp_min(target, 0))
            # print('target.shape:', target_one_hot.shape)

            return self._base_forward(output, target_one_hot, valid_mask)