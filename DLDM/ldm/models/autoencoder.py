import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder, SimpleMaskDecoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config, normalize_intensity
import copy, random
import numpy as np

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.denorm(x)
            xrec = self.denorm(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.denorm(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.save_hyperparameters() # for tracking hyperparameter in wandb
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig) # encoder latent b 12 64 64
        decoder_ddconfig = copy.deepcopy(ddconfig) 
        decoder_ddconfig['z_channels'] = 4 # b 4 64 64
        decoder_ddconfig['in_channels'] = 4
        self.decoder = Decoder(**decoder_ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.post_quant_conv = torch.nn.Conv2d(decoder_ddconfig["z_channels"], decoder_ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.class_idx = [1, 2, 3, 4, 5]

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, random_idx=None): # random_idx -> [1, 2, 3, 4, 5] = T1, T2, FL, WB, BB
        if random_idx is not None: # when b, 12, 64, 64 is input, make it b, 4, 64, 64.
            structure_latent = z[:, :2, :, :]
            style_latent = z[:, random_idx*2:random_idx*2+1, :, :]
            z = torch.cat([structure_latent, style_latent], dim=1) # b, 4, 64, 64
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, random_idx, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        # print('z.shape in forward:', z.shape) # 6, 6, 64, 64
        dec = self.decode(z, random_idx)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key) # b, 5, 256, 256

        post_list = []
        struct_dict = {}
        style_dict = {}
        posterior_dict = {}

        # This loop is for saving struct and style with data index in minibatch. later used for Rand Mix
        for b in range(inputs.shape[0]):
            posterior = self.encode(inputs[b].unsqueeze(0))
            z = posterior.sample()    # 1, 12, 64, 64
            struct = z[:, :2, :, :]   # 1,  2, 64, 64
            style = z[:, 2:, :, :]    # 1, 10, 64, 64
            struct_dict[b] = struct   # data index in minibatch : structure vector [1, 2, 64, 64]
            style_dict[b] = style     # data index in minibatch : style vector [1, 10, 64, 64]
            posterior_dict[b] = posterior

        # define mixed structure and style inputs
        posterior = self.encode(inputs) # encoding again to get full minibatch latent z. (encoding twice is not appropriate. will be updated later)
        z = posterior.sample() # b, 12, 64, 64    

        # Rand Mix
        randomize_batch_idx = [x for x in range(inputs.shape[0])]
        random.shuffle(randomize_batch_idx) # if batch size is 6, randomize order ex. 3, 4, 2, 1, 5, 0
        for i, j in enumerate(randomize_batch_idx):
            style = style_dict[j]
            z[i, 2:, :, :] = style # 1, 12, 64, 64 <- structure and style is now mixed!
        
        # LOSS is calculated with random 3 sequence averaged
        b = inputs.shape[0]
        # three_random_idx = random.sample(self.class_idx, 3) # class_idx = [1, 2, 3, 4, 5] meaning [T1, T2, FL, WB, BB] respectively
        for i, idx in enumerate(self.class_idx): # ex. 1, 3, 4 = T1, FL, WB
            if i == 0:
                reconstructions = self.decode(z, idx) # if idx is 3, decode z with [5, 6] channel -> 6, 1, 256, 256
                single_inputs = inputs[:, idx-1, :, :].unsqueeze(1) # if idx is 3, get corresponding flair -> 6, 1, 256, 256
                post_list.append(posterior_dict[idx])
            else:
                temp_reconstructions = self.decode(z, idx) # if idx is 1, decode z with [1, 2] channel -> 6, 1, 256, 256
                temp_single_inputs = inputs[:, idx-1, :, :].unsqueeze(1) # if idx is 1, get corresponding T1 -> 6, 1, 256, 256
                reconstructions = torch.cat([reconstructions, temp_reconstructions], dim=1)
                single_inputs = torch.cat([single_inputs, temp_single_inputs], dim=1)
                post_list.append(posterior_dict[idx-1])

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(single_inputs, reconstructions, post_list, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(single_inputs, reconstructions, post_list, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key) # b, 5, 256, 256

        randomize_batch_idx = [x for x in range(inputs.shape[0])]
        random.shuffle(randomize_batch_idx)

        post_list = []
        struct_dict = {}
        style_dict = {}
        posterior_dict = {}

        # This loop is for saving struct and style with data index in minibatch. later used for Rand Mix
        for b in range(inputs.shape[0]):
            posterior = self.encode(inputs[b].unsqueeze(0))
            z = posterior.sample()    # 1, 12, 64, 64
            struct = z[:, :2, :, :]   # 1,  2, 64, 64
            style = z[:, 2:, :, :]    # 1, 10, 64, 64
            struct_dict[b] = struct   # data index in minibatch : structure vector [1, 2, 64, 64]
            style_dict[b] = style     # data index in minibatch : style vector [1, 10, 64, 64]
            posterior_dict[b] = posterior

        # define mixed structure and style inputs
        posterior = self.encode(inputs) # encoding again to get full minibatch latent z. (encoding twice is not appropriate. will be updated later)
        z = posterior.sample() # b, 12, 64, 64    

        # Rand Mix
        randomize_batch_idx = [x for x in range(inputs.shape[0])]
        random.shuffle(randomize_batch_idx) # if batch size is 6, randomize order ex. 3, 4, 2, 1, 5, 0
        for i, j in enumerate(randomize_batch_idx):
            style = style_dict[j]
            z[i, 2:, :, :] = style # 1, 12, 64, 64 <- structure and style is now mixed!
            
        # LOSS is calculated with random 3 sequence averaged
        b = inputs.shape[0]
        # three_random_idx = random.sample(self.class_idx, 3) # class_idx = [1, 2, 3, 4, 5] = T1, T2, FL, WB, BB
        for i, idx in enumerate(self.class_idx): # ex. 1, 3, 4 = T1, FL, WB
            if i == 0:
                reconstructions = self.decode(z, idx) # if idx is 3, decode z with [5, 6] channel -> 6, 1, 256, 256
                single_inputs = inputs[:, idx-1, :, :].unsqueeze(1) # if idx is 3, get corresponding flair -> 6, 1, 256, 256
                post_list.append(posterior_dict[idx])
            else:
                temp_reconstructions = self.decode(z, idx) # if idx is 1, decode z with [1, 2] channel -> 6, 1, 256, 256
                temp_single_inputs = inputs[:, idx-1, :, :].unsqueeze(1) # if idx is 1, get corresponding T1 -> 6, 1, 256, 256
                reconstructions = torch.cat([reconstructions, temp_reconstructions], dim=1)
                single_inputs = torch.cat([single_inputs, temp_single_inputs], dim=1)
                post_list.append(posterior_dict[idx-1])

        aeloss, log_dict_ae = self.loss(single_inputs, reconstructions, post_list, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(single_inputs, reconstructions, post_list, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device) # b, 5, 256, 256
        b = x.shape[0]
        batch_idx_list = [i for i in range(b)]
        sampled_batch_idx = random.sample(batch_idx_list, 2)

        x_batch_0 = x[sampled_batch_idx[0]].unsqueeze(0) # 1, 5, 256, 256
        x_batch_1 = x[sampled_batch_idx[1]].unsqueeze(0) # 1, 5, 256, 256
        if not only_inputs:
            random_idx = random.randint(1, 5)
            xrec, posterior = self(x, random_idx)

            xrec_0_0, _ = self(x_batch_0, 1) # 1, 1, 256, 256
            xrec_1_0, _ = self(x_batch_0, 2)
            xrec_2_0, _ = self(x_batch_0, 3)
            xrec_3_0, _ = self(x_batch_0, 4)
            xrec_4_0, _ = self(x_batch_0, 5)

            xrec_0_1, _ = self(x_batch_1, 1) # 1, 1, 256, 256
            xrec_1_1, _ = self(x_batch_1, 2)
            xrec_2_1, _ = self(x_batch_1, 3)
            xrec_3_1, _ = self(x_batch_1, 4)
            xrec_4_1, _ = self(x_batch_1, 5)

            # mix z0 structure and z1 style
            posterior0 = self.encode(x_batch_0)
            z0 = posterior0.sample() # 1, 12, 64, 64
            posterior1 = self.encode(x_batch_1)
            z1 = posterior1.sample() # 1, 12, 64, 64
            z0[:, 2:, :, :] = z1[:, 2:, :, :]
            xrec_0_mix = self.decode(z0, 1)
            xrec_1_mix = self.decode(z0, 2)
            xrec_2_mix = self.decode(z0, 3)
            xrec_3_mix = self.decode(z0, 4)
            xrec_4_mix = self.decode(z0, 5)

            if x.shape[1] > 3 and x_batch_0.shape[1] > 3:
                x = self.denorm(x)
                x_batch_0 = self.denorm(x_batch_0)
                x_batch_1 = self.denorm(x_batch_1)

                b = x.shape[0]
                for i in range(x.shape[1]):
                    if i == 0:
                        temp = x[:, i, :, :].unsqueeze(1)
                    else:
                        temp = torch.cat([temp, x[:, i, :, :].unsqueeze(1)], dim=3)
                x = temp

                b = x_batch_0.shape[0]
                for i in range(x_batch_0.shape[1]):
                    if i == 0:
                        temp = x_batch_0[:, i, :, :].unsqueeze(1)
                    else:
                        temp = torch.cat([temp, x_batch_0[:, i, :, :].unsqueeze(1)], dim=3)
                x_batch_0 = temp

                b = x_batch_1.shape[0]
                for i in range(x_batch_1.shape[1]):
                    if i == 0:
                        temp = x_batch_1[:, i, :, :].unsqueeze(1)
                    else:
                        temp = torch.cat([temp, x_batch_1[:, i, :, :].unsqueeze(1)], dim=3)
                x_batch_1 = temp

                xrec = self.denorm(xrec) # b, 1, 256, 256
                xrec_0_0 = self.denorm(xrec_0_0) # 1, 1, 256, 256
                xrec_1_0 = self.denorm(xrec_1_0)
                xrec_2_0 = self.denorm(xrec_2_0)
                xrec_3_0 = self.denorm(xrec_3_0)
                xrec_4_0 = self.denorm(xrec_4_0)

                xrec_0_1 = self.denorm(xrec_0_1) # 1, 1, 256, 256
                xrec_1_1 = self.denorm(xrec_1_1)
                xrec_2_1 = self.denorm(xrec_2_1)
                xrec_3_1 = self.denorm(xrec_3_1)
                xrec_4_1 = self.denorm(xrec_4_1)

                xrec_0_mix = self.denorm(xrec_0_mix) # 1, 1, 256, 256
                xrec_1_mix = self.denorm(xrec_1_mix)
                xrec_2_mix = self.denorm(xrec_2_mix)
                xrec_3_mix = self.denorm(xrec_3_mix)
                xrec_4_mix = self.denorm(xrec_4_mix)

                xrec_0 = torch.cat([xrec_0_0, xrec_1_0, xrec_2_0, xrec_3_0, xrec_4_0], dim=0) # 1, 1, 256, 1258
                xrec_1 = torch.cat([xrec_0_1, xrec_1_1, xrec_2_1, xrec_3_1, xrec_4_1], dim=0) # 1, 1, 256, 1258
                xrec_multi_mix = torch.cat([xrec_0_mix, xrec_1_mix, xrec_2_mix, xrec_3_mix, xrec_4_mix], dim=0) # 1, 1, 256, 1258

            log["samples_{}".format(random_idx)] = self.decode(torch.randn_like(posterior.sample()), random_idx)
            log["reconstructions_{}".format(random_idx)] = xrec
            log["xrec_0"] = xrec_0
            log["xrec_1"] = xrec_1
            log["0_1_mixed_rec"] = xrec_multi_mix

        log["inputs"] = x
        log['x_batch_0'] = x_batch_0
        log['x_batch_1'] = x_batch_1
        return log

    def denorm(self, x):
        # assert self.image_key == "segmentation"
        # if not hasattr(self, "colorize"):
        #     self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        # x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

class DAE(pl.LightningModule):
    def __init__(self,
                 enc_config,
                 dec_config,
                 mask_dec_config,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.save_hyperparameters() # for tracking hyperparameter in wandb
        self.image_key = image_key
        self.encoder = Encoder(**enc_config) # encoder latent b 7 64 64 -> b 2 64 64: struct, b 4 64 64: style, b 1 64 64: mask
        self.decoder = Decoder(**dec_config)
        self.mask_decoder = SimpleMaskDecoder(**mask_dec_config)
        self.loss = instantiate_from_config(lossconfig)
        self.double_z = enc_config["double_z"]
        assert enc_config["double_z"]
        assert dec_config["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*enc_config["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(dec_config["z_channels"], dec_config["z_channels"], 1)

        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.class_idx = [1, 2, 3, 4] # T2, FLAIR, WB, BB

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # print('1:', x.shape) # b, 5, 256, 256
        h = self.encoder(x)
        # print('2:', h.shape) # b, 14, 64, 64
        moments = self.quant_conv(h)
        # print('3:', moments.shape) # b, 14, 64, 64
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        # print('4:', z.shape) # 
        post_z = self.post_quant_conv(z)
        # print('5:', post_z.shape)
        dec = self.decoder(post_z)
        return dec
    
    def mask_decode(self, mask):
        mask = self.mask_decoder(mask)
        return mask
    
    def z_selector(self, z, idx, struct_dim=2, style_dim=1):
        # idx -> 0, 1, 2, 3, 4 = T2, FLAIR, WB, BB, MASK
        
        struct = z[:, :struct_dim, :, :]
        # print('struct:', struct.shape)
        style = z[:, struct_dim+style_dim*idx:struct_dim+style_dim*(idx+1), :, :]
        # print('stye:', style.shape)
        merged_z = torch.cat([struct, style], dim=1)
        # print('merged_z:', merged_z.shape)
        return merged_z
    
    def rand_mix(self, z, struct_dim=2):
        struct_dict = {}
        style_dict = {}
        # posterior_dict = {}
        for b in range(z.shape[0]):
            # struct = z[:, :struct_dim, :, :]   # 1, 2, 64, 64
            style = z[b, struct_dim:, :, :]    # 1, 5, 64, 64
            # struct_dict[b] = struct   # data index in minibatch : structure vector [1, 2, 64, 64]
            style_dict[b] = style     # data index in minibatch : style vector [1, 10, 64, 64]
            # posterior_dict[b] = posterior
        # structure = z[:, :struct_dim, :, :]
        if z.shape[0] == 2:
            randomize_batch_idx = [1, 0]
        else:
            randomize_batch_idx = [x for x in range(z.shape[0])]
            random.shuffle(randomize_batch_idx) # if batch size is 6, randomize order ex. 3, 4, 2, 1, 5, 0
        for i, j in enumerate(randomize_batch_idx):
            style = style_dict[j]
            z[i, struct_dim:, :, :] = style # 1, 12, 64, 64 <- structure and style is now mixed!
        return z

    def forward(self, input, idx, mix=False, sample_posterior=True):
        # input shape -> b, 5, 256, 256
        posterior = self.encode(input) # b, 14, 64, 64
        if sample_posterior:
            z = posterior.sample()   # b, 7, 64, 64
        else:
            z = posterior.mode()
        if mix == True:
            z = self.rand_mix(z)
        z = self.z_selector(z, idx) # b 3 64 64
        dec = self.decode(z) # b 3 64 64 -> b 1 256 256
        if idx == 4:
            dec = self.mask_decode(dec)
        return dec

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key) # b, 5, 256, 256 -> 4 sequences and a mask = 5
        posterior = self.encode(inputs) # moments is matrix for mean and std. channel is 14 because of double z (double z is for reducing uncertainty)
        z = posterior.sample() # b, 7, 64, 64
        z = self.rand_mix(z)

        post_list = []
        ch = inputs.shape[1]
        for c in range(ch):
            try:
                temp_recon = self.decode(self.z_selector(z, c))
                recon = torch.cat([recon, temp_recon], dim=1)
                temp_single_input = inputs[:, c, :, :].unsqueeze(1)
                single_input = torch.cat([single_input, temp_single_input], dim=1)
            except:
                recon = self.decode(self.z_selector(z, c))
                single_input = inputs[:, c, :, :].unsqueeze(1)
            post_list.append(posterior)

        ####################################################
        # img_real = single_input[:, :4, :, :]
        mask_real = single_input[:, 4, :, :].unsqueeze(1)
        
        # img_fake = recon[:, :4, :, :]
        mask_fake = recon[:, 4, :, :].unsqueeze(1)
        mask_fake = self.mask_decode(mask_fake)

        # print('real:', mask_real.max(), mask_real.min())
        # print('fake:', mask_fake.max(), mask_fake.min())


        # img_post = post_list[:4]
        # mask_post = [post_list[4]]
        ####################################################
        # print('optimizer_idx:', optimizer_idx)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(single_input, recon, post_list, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", mask=False)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            # print('aeloss:', aeloss)
            return aeloss
    
        if optimizer_idx == 1:
            # train the mask decoder
            mask_loss, log_dict_mask = self.loss(mask_real, mask_fake, post_list, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split='train', mask=True)
            self.log("maskloss", mask_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_mask, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            # print('mask_loss', mask_loss)
            return mask_loss

        # # train encoder+decoder+logvar
        # if batch_idx%2==0: # img loss
        #     img_aeloss, img_log_dict_ae = self.loss(img_real, img_fake, img_post, 0, self.global_step,
        #                                 last_layer=self.get_last_layer(), split="train", mask=False)
        #     self.log("img_aeloss", img_aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     self.log_dict(img_log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        #     return img_aeloss
        # else: # mask loss
        #     mask_aeloss, mask_log_dict_ae = self.loss(mask_real, mask_fake, mask_post, 0, self.global_step,
        #                                 last_layer=self.get_last_layer(), split="train", mask=True)
        #     self.log("mask_aeloss", mask_aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #     self.log_dict(mask_log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        #     return mask_aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key) # b, 5, 256, 256 -> 4 sequences and a mask = 5
        posterior = self.encode(inputs) # moments is matrix for mean and std. channel is 14 because of double z (double z is for reducing uncertainty)
        z = posterior.sample() # b, 7, 64, 64
        z = self.rand_mix(z)

        post_list = []
        ch = inputs.shape[1]
        for c in range(ch):
            try:
                temp_recon = self.decode(self.z_selector(z, c))
                recon = torch.cat([recon, temp_recon], dim=1)
                temp_single_input = inputs[:, c, :, :].unsqueeze(1)
                single_input = torch.cat([single_input, temp_single_input], dim=1)
            except:
                recon = self.decode(self.z_selector(z, c))
                single_input = inputs[:, c, :, :].unsqueeze(1)
            post_list.append(posterior)

        ####################################################
        # img_real = single_input[:, :4, :, :]
        mask_real = single_input[:, 4, :, :].unsqueeze(1)

        # img_fake = recon[:, :4, :, :]
        mask_fake = recon[:, 4, :, :].unsqueeze(1)
        mask_fake = self.mask_decode(mask_fake)

        # img_post = post_list[:4]
        # mask_post = [post_list[4]]
        ####################################################

        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(single_input, recon, post_list, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val", mask=False)

        mask_loss, log_dict_mask = self.loss(mask_real, mask_fake, post_list, 1, self.global_step,
                                             last_layer=self.get_last_layer(), split='val', mask=True)
        
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log("val/mask_loss", log_dict_mask["val/mask_loss"])
        self.log_dict(log_dict_mask)
        
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_mask = torch.optim.Adam(list(self.mask_decoder.parameters()), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_mask], []

    def get_last_layer(self):
        return self.mask_decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device) # b, 5, 256, 256
        
        batch_size, ch, _, _ = x.shape

        b_idx_list = [i for i in range(batch_size)] # 0, 1, 2, ..., b
        rand_b_idx = random.sample(b_idx_list, 2) # 3, 6

        x_batch_0 = x[rand_b_idx[0]].unsqueeze(0) # 1, 5, 256, 256
        x_batch_1 = x[rand_b_idx[1]].unsqueeze(0) # 1, 5, 256, 256

        cat = torch.cat([x_batch_0, x_batch_1], dim=0) # 2, 5, 256, 256
        batch_size = cat.shape[0] # 2
        
        if not only_inputs:
            # Non-mixed
            for c in range(ch): # 0, 1, 2, 3, 4 = T2, FLAIR, WB, BB, MASK
                try:
                    new_rec = self(cat, c) #######################
                    rec = torch.cat([rec, new_rec], dim=1) # final rec.shape = 2, 5, 256, 256
                except:
                    rec = self(cat, c) # 2, 1, 256, 256
            
            # Mixed
            for c in range(ch): # 0, 1, 2, 3, 4 = T2, FLAIR, WB, BB, MASK
                try:
                    new_mix_rec = self(cat, c, mix=True)
                    mix_rec = torch.cat([mix_rec, new_mix_rec], dim=1) # final rec.shape = 2, 5, 256, 256
                except:
                    mix_rec = self(cat, c, mix=True) # 2, 1, 256, 256

            if x.shape[1]>1: # slice denormalize
                b = x.shape[0]
                for c in range(x.shape[1]):
                    if c == 0:
                        temp = x[:, c, :, :].unsqueeze(1)
                    else:
                        temp = torch.cat([temp, x[:, c, :, :].unsqueeze(1)], dim=3)
                x = temp

                for b in range(batch_size):
                    for c in range(ch):
                        if c != ch-1: #img
                            # print('before_norm img:', rec[b, c, :, :].max(), rec[b, c, :, :].min())
                            cat[b, c, :, :] = self.denorm(cat[b, c, :, :])
                            rec[b, c, :, :] = self.denorm(rec[b, c, :, :])
                            mix_rec[b, c, :, :] = self.denorm(mix_rec[b, c, :, :])
                            # print('after_norm img:', rec[b, c, :, :].max(), rec[b, c, :, :].min())
                        else: #mask
                            # print('before_norm mask:', rec[b, c, :, :].max(), rec[b, c, :, :].min())
                            cat[b, c, :, :] = self.multi_mask_vis(self.denorm(cat[b, c, :, :]))
                            rec[b, c, :, :] = self.multi_mask_vis(self.denorm(rec[b, c, :, :]))
                            mix_rec[b, c, :, :] = self.multi_mask_vis(self.denorm(mix_rec[b, c, :, :]))
                            # print('after_norm mask:', rec[b, c, :, :].max(), rec[b, c, :, :].min())

                for c in range(ch):
                    try:
                        real_0 = torch.cat([real_0, cat[0][c]], dim=1)
                        real_1 = torch.cat([real_1, cat[1][c]], dim=1)
                        rec_0 = torch.cat([rec_0, rec[0][c]], dim=1)
                        rec_1 = torch.cat([rec_1, rec[1][c]], dim=1)
                        mix_0_1 = torch.cat([mix_0_1, mix_rec[1][c]], dim=1)
                    except:
                        real_0 = cat[0][c]
                        real_1 = cat[1][c]
                        rec_0 = rec[0][c]
                        rec_1 = rec[1][c]
                        mix_0_1 = mix_rec[1][c] 
                        
            log["rec_img_batch_0"] = rec_0.unsqueeze(0).unsqueeze(0)
            log["rec_img_batch_1"] = rec_1.unsqueeze(0).unsqueeze(0)
            log["rec_img_mixed"] = mix_0_1.unsqueeze(0).unsqueeze(0)
        
        log["inputs"] = x
        log['x_batch_0'] = real_0.unsqueeze(0).unsqueeze(0)
        log['x_batch_1'] = real_1.unsqueeze(0).unsqueeze(0)
        return log

    def denorm(self, x): # For visualizing 2d matrix makes -1 ~ 1
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

    def cliping(self, x): # For visualizing 2d matrix
        # cliping
        x=x.cpu().numpy()
        background = np.mean(x[:5][:5])
        if background>x.min():
            x[x<background] = background
        x = torch.from_numpy(x)
        return x

    def multi_mask_vis(self, x): # For visualizing 2d matrix
        x=x.cpu().numpy()
        x = np.where(x > 0.75, 1., x) # necro
        x = np.where((x > 0.25) & (x < 0.75), 0.5, x) # enhance
        x = np.where((x > -0.25) & (x < 0.25), 0., x) # edema
        x = np.where((x > -0.75) & (x < -0.25), -0.5, x) # meta
        x = np.where(x < -0.75, -1., x) # background
        x = torch.from_numpy(x)
        return x
    
    def mask_channel_arrange(self, mask): # single channel mask to multi channel mask
        # mask.shape = b, 1, 256, 256
        batch, ch, _, _ = mask.shape
        multi_mask = torch.zeros(batch, 4, 256, 256) # b, 4, 256, 256
        for b in range(batch):
            multi_mask[b, 0, :, :] = torch.where(mask[b] > 0.75, mask[b], -1.) # necro
            multi_mask[b, 1, :, :] = torch.where((mask[b] > 0.25) & (mask[b] < 0.75), mask[b], -1.) # enhance
            multi_mask[b, 2, :, :] = torch.where((mask[b] > -0.25) & (mask[b] < 0.25), mask[b], -1.) # edema
            multi_mask[b, 3, :, :] = torch.where((mask[b] > -0.75) & (mask[b] < -0.25), mask[b], -1.) # meta
        # print('temp:', temp.shape)
        return multi_mask

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
