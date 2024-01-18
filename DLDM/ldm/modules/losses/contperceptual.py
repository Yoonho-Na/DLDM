import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?



class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):

        # average over 3 sequences [b 3 256 256]
        rec_loss = 0
        p_loss = 0
        for i in range(inputs.shape[1]):
            rec_loss += torch.abs(inputs[:, i, :, :].unsqueeze(1).contiguous() - reconstructions[:, i, :, :].unsqueeze(1).contiguous())
            if self.perceptual_weight > 0:
                p_loss += self.perceptual_loss(inputs[:, i, :, :].unsqueeze(1).contiguous(), reconstructions[:, i, :, :].unsqueeze(1).contiguous())
        rec_loss = (0.3333)*rec_loss + self.perceptual_weight*(0.3333)*p_loss
        
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        
        for i in range(inputs.shape[1]):
            if i == 0:
                kl_loss = posteriors[i].kl()
            else:
                kl_loss = torch.add(kl_loss, posteriors[i].kl())
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            # print('g_loss:', g_loss)
            # print('nll_loss:', nll_loss)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
















































class LPIPS_DAE_loss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0, perceptual_weight=1.0):

        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.mask_weight = 1 # 0.5
        self.mask_loss = nn.CrossEntropyLoss()
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", mask=False,
                weights=None):

        rec_loss = 0
        p_loss = 0
        mask_loss = 0
        ch = inputs.shape[1]

        if mask == False: # image
            for i in range(ch):
                rec_loss += torch.abs(inputs[:, i, :, :].unsqueeze(1).contiguous() - reconstructions[:, i, :, :].unsqueeze(1).contiguous())
                if self.perceptual_weight > 0:
                    p_loss += self.perceptual_loss(inputs[:, i, :, :].unsqueeze(1).contiguous(), reconstructions[:, i, :, :].unsqueeze(1).contiguous())
            rec_loss = (1./ch)*rec_loss + self.perceptual_weight*(1./ch)*p_loss
        else:             # mask
            for i in range(ch):
                rec_loss += torch.abs(inputs[:, i, :, :].unsqueeze(1).contiguous() - reconstructions[:, i, :, :].unsqueeze(1).contiguous())
                if self.mask_weight > 0:
                    mask_loss += self.mask_loss(reconstructions[:, i, :, :].unsqueeze(1).contiguous(), inputs[:, i, :, :].unsqueeze(1).contiguous())
            rec_loss = rec_loss + self.mask_weight*mask_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        
        for i in range(ch):
            if i == 0:
                kl_loss = posteriors[i].kl()
            else:
                kl_loss = torch.add(kl_loss, posteriors[i].kl())
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss

        if mask == False: # image
            log = {"{}/total_img_loss".format(split): loss.clone().detach().mean(), "{}/img_logvar".format(split): self.logvar.detach(),
                    "{}/kl_img_loss".format(split): kl_loss.detach().mean(), "{}/nll_img_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_img_loss".format(split): rec_loss.detach().mean(),
                    }
        else:             # mask
            log = {"{}/total_mask_loss".format(split): loss.clone().detach().mean(), "{}/mask_logvar".format(split): self.logvar.detach(),
                    "{}/kl_mask_loss".format(split): kl_loss.detach().mean(), "{}/mask_nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_mask_loss".format(split): rec_loss.detach().mean(),
                    }
        return loss, log

    def dice_loss(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        smooth = 1.0
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        return 1. - dsc

class LPIPS_DAE_integ_loss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0, perceptual_weight=1.0):

        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.mask_weight = 1 # 0.5
        # self.mask_loss = nn.CrossEntropyLoss()
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.focal_loss = torch.hub.load(
                                        'adeelh/pytorch-multi-class-focal-loss',
                                        model='FocalLoss',
                                        alpha=torch.tensor([.75, .25]),
                                        gamma=2,
                                        reduction='mean',
                                        force_reload=False
                                        )

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", mask=False,
                weights=None):

        rec_loss = 0
        p_loss = 0
        ch = inputs.shape[1]

        if optimizer_idx == 0:

            for i in range(ch):
                rec_loss += torch.abs(inputs[:, i, :, :].unsqueeze(1).contiguous() - reconstructions[:, i, :, :].unsqueeze(1).contiguous())
                if self.perceptual_weight > 0:
                    p_loss += self.perceptual_loss(inputs[:, i, :, :].unsqueeze(1).contiguous(), reconstructions[:, i, :, :].unsqueeze(1).contiguous())
            rec_loss = (1./ch)*rec_loss + self.perceptual_weight*(1./ch)*p_loss

            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights*nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            
            for i in range(ch):
                if i == 0:
                    kl_loss = posteriors[i].kl()
                else:
                    kl_loss = torch.add(kl_loss, posteriors[i].kl())
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            loss = weighted_nll_loss + self.kl_weight * kl_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                        "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                        "{}/rec_loss".format(split): rec_loss.detach().mean(),}
            return loss, log

        if optimizer_idx == 1:
            for i in range(ch):
                rec_loss += torch.abs(inputs[:, i, :, :].unsqueeze(1).contiguous() - reconstructions[:, i, :, :].unsqueeze(1).contiguous())
            rec_loss = (1./ch)*rec_loss

            inputs = mask_channel_arrange(inputs)
            reconstructions = mask_channel_arrange(reconstructions)
            multi_dice_loss = dice_loss(inputs.contiguous(), reconstructions.contiguous())

            mask_loss = 0.5 * rec_loss.mean() + 0.5 * multi_dice_loss # combination of L1 loss and dice with half weights
            log = {"{}/mask_loss".format(split): mask_loss.detach().mean()}
            return mask_loss, log
        

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1. - fn(input, target, reduce_batch_first=True)

def mask_channel_arrange(mask):
    # mask.shape = b, 1, 256, 256
    batch, ch, _, _ = mask.shape
    temp = torch.zeros(batch, 4, 256, 256) # b, 4, 256, 256
    for b in range(batch):
        temp[b, 0, :, :] = torch.where(mask[b] > 0.75, mask[b], -1.) # necro
        temp[b, 1, :, :] = torch.where((mask[b] > 0.25) & (mask[b] < 0.75), mask[b], -1.) # enhance
        temp[b, 2, :, :] = torch.where((mask[b] > -0.25) & (mask[b] < 0.25), mask[b], -1.) # edema
        temp[b, 3, :, :] = torch.where((mask[b] > -0.75) & (mask[b] < -0.25), mask[b], -1.) # meta
    # print('temp:', temp.shape)
    return temp
