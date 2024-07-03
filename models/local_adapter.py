import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel, 
    TimestepBlock, 
    TimestepEmbedSequential, 
    ResBlock, 
    Downsample, 
    AttentionBlock
)
from ldm.modules.attention import SpatialTransformer
from ldm.util import exists


def layer_norm(tensor, drop=0.5, eps=1e-6):
    mean = tensor.mean(dim=(1,2)).squeeze()
    std = tensor.std(dim=(1,2)).squeeze()
    var = tensor.var(dim=(1,2))
    tensor = (tensor-mean) / (var+eps) ** 0.5
    neg = (tensor * (tensor < 0).float()).abs().sum() / (tensor<0).float().sum()
    pos = (tensor * (tensor > 0).float()).abs().sum() / (tensor>0).float().sum()


class LocalTimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, local_control=None, content_control=None, color_control=None, content_w=1.0, color_w=1.0):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, content_control, color_control, content_w, color_w)
            elif isinstance(layer, LocalResBlock):
                x = layer(x, emb, local_control)
            else:
                x = layer(x)
        return x


class FDN(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3
        pw = ks // 2
        self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=False)
        self.conv_gamma = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, local_features):
        normalized = self.param_free_norm(x)
        assert local_features.size()[2:] == x.size()[2:]
        gamma = self.conv_gamma(local_features)
        beta = self.conv_beta(local_features)
        out = normalized * (1 + gamma) + beta
        return out


class LocalResBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        use_checkpoint=False,
        inject_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.norm_in = FDN(channels, inject_channels)
        self.norm_out = FDN(self.out_channels, inject_channels)

        self.in_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, local_conditions):
        return checkpoint(
            self._forward, (x, emb, local_conditions), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, local_conditions):
        local_conditions = F.interpolate(local_conditions, x.shape[-2:], mode="bilinear")
        h = self.norm_in(x, local_conditions)
        h = self.in_layers(h)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        h = h + emb_out
        h = self.norm_out(h, local_conditions)
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h


class LocalAdapter(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            local_channels,
            inject_channels,
            inject_layers,
            query_channels,
            query_layers,
            query_scales,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.inject_layers = inject_layers
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        self.query_channels = query_channels
        self.query_layers = query_layers
        self.query_scales = query_scales
        visual_projs = []
        for query_channel, inject_channel in zip(query_channels, inject_channels):
            layer_proj = zero_module(linear(query_channel, inject_channel))
            visual_projs.append(layer_proj)
        self.visual_projs = nn.ModuleList(visual_projs)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                LocalTimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                if (1 + 3*level + nr) in self.inject_layers:
                    layers = [
                        LocalResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            inject_channels=inject_channels[level],
                        )
                    ]
                else:
                    layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(LocalTimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    LocalTimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = LocalTimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return LocalTimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def extract_local_features(self, q_former, text, local_conditions):
        # extract local features
        bs, chn, h, w = local_conditions.shape
        n = chn // 3
        image_features_frozen, image_atts = q_former.forward_visual_encoder(local_conditions.view(bs * n, 3, h, w))
        bs_n, seq_len, v_chn = image_features_frozen[0].shape 

        # with pos embed
        image_features_frozen = [q_former.crossattn_embeddings(image_feat) for image_feat in image_features_frozen]

        # image_features_frozen: [bs * n, seq_len, c]
        image_features_frozen = [image_feat.view(bs, n*seq_len, v_chn) for image_feat in image_features_frozen]
        image_atts = [image_att.view(bs, -1) for image_att in image_atts]

        local_embeddings = q_former.forward_qformer(text, image_features_frozen, image_atts)

        # process qformer features
        local_features = []
        for lvl, scale_factor, visual_proj in zip(self.query_layers, self.query_scales, self.visual_projs):
            local_emb = local_embeddings[lvl]
            _, seq_len, ndim = local_emb.shape
            l = int(seq_len ** 0.5)
            local_emb = F.interpolate(local_emb.transpose(1,2).view(bs, -1, l, l), None, scale_factor=scale_factor, mode="bilinear")
            local_emb = visual_proj(local_emb.transpose(1,2).transpose(2,3).flatten(1,2))
            local_emb = local_emb.view(bs, int(l*scale_factor), int(l*scale_factor), -1).transpose(2,3).transpose(1,2)
            local_features.append(local_emb)
        return local_features

    def forward(self, x, timesteps, context, local_features, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        outs = []
        h = x.type(self.dtype)
        for layer_idx, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            if layer_idx in self.inject_layers:
                h = module(h, emb, context, local_control=local_features[self.inject_layers.index(layer_idx)])
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class LocalControlUNetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, local_control=None, content_control=None, color_control=None, local_w=1.0, content_w=1.0, color_w=1.0, **kwargs):
        hs = []

        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context, content_control=content_control, color_control=color_control, content_w=content_w, color_w=color_w)
                hs.append(h)
            h = self.middle_block(h, emb, context, content_control=content_control, color_control=color_control, content_w=content_w, color_w=color_w)

        h += local_w * local_control.pop()

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop() + local_w * local_control.pop()], dim=1)
            h = module(h, emb, context, content_control=content_control, color_control=color_control, content_w=content_w, color_w=color_w)

        h = h.type(x.dtype)
        return self.out(h)
