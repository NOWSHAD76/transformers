import copy
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union


from ...utils import auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ..gemma3.configuration_gemma3 import Gemma3Config, Gemma3TextConfig
from ..qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniToken2WavConfig,
    Qwen2_5OmniConfig,
)
from ...configuration_utils import PretrainedConfig
from ..gemma3.modeling_gemma3 import (
    Gemma3PreTrainedModel,
    Gemma3ForConditionalGeneration,
    Gemma3CausalLMOutputWithPast,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniTalkerForConditionalGeneration,
    Qwen2_5OmniToken2WavModel,
    Qwen2_5OmniPreTrainedModel,
    Qwen2_5OmniTalkerCausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...modeling_outputs import BaseModelOutput, ModelOutput
import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm
from ...generation import GenerationMixin
from ...utils import (
    auto_docstring,
    check_torch_load_is_safe,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from ...utils.hub import cached_file
from loguru import logger as logger_v2

logger = logging.get_logger(__name__)


class Gemma3ThinkerConfig(Gemma3Config):
    # model_type = "gemma3"
    pass


class Gemma3Config(Gemma3Config):
    pass


class Gemma3TextConfig(Gemma3TextConfig):
    pass


class Gemma3TalkerConfig(Qwen2_5OmniTalkerConfig):
    # model_type = "qwen2_5_omni_talker"
    pass


class Gemma3Token2WavConfig(Qwen2_5OmniToken2WavConfig):
    # model_type = "qwen2_5_omni_token2wav"
    pass


class Gemma3WithTalkerConfig(PretrainedConfig):
    model_type = "gemma3_with_talker"
    sub_configs = {
        "thinker_config": Gemma3ThinkerConfig,
        "talker_config": Gemma3TalkerConfig,
        "token2wav_config": Gemma3Token2WavConfig,
    }

    def __init__(
        self,
        thinker_config: Optional[Union[Gemma3ThinkerConfig, Dict[str, Any]]] = None,
        talker_config: Optional[Union[Gemma3TalkerConfig, Dict[str, Any]]] = None,
        token2wav_config: Optional[Union[Gemma3Token2WavConfig, Dict[str, Any]]] = None,
        enable_audio_output: bool = True,
        **kwargs,
    ):
        if thinker_config is None:
            thinker_config = Gemma3ThinkerConfig()
            logger.info(
                "thinker_config is None, using default Gemma3ThinkerConfig config."
            )
        elif isinstance(thinker_config, dict):
            thinker_config = Gemma3ThinkerConfig(**thinker_config)

        if isinstance(talker_config, dict):
            talker_config = Gemma3TalkerConfig(**talker_config)
        elif talker_config is None:
            talker_config = Gemma3TalkerConfig()
            logger.info(
                "talker_config is None, using default Gemma3TalkerConfig config."
            )
        if isinstance(token2wav_config, dict):
            token2wav_config = Gemma3Token2WavConfig(**token2wav_config)
        elif token2wav_config is None:
            token2wav_config = Gemma3Token2WavConfig()
            logger.info(
                "token2wav_config is None, using default Gemma3Token2WavConfig config."
            )
        self.thinker_config = thinker_config
        self.talker_config = talker_config
        self.token2wav_config = token2wav_config
        self.enable_audio_output = enable_audio_output

        super().__init__(**kwargs)

    def get_text_config(self, decoder=False) -> "PretrainedConfig":
        """
        Returns the config that is meant to be used with text IO. On most models, it is the original config instance
        itself. On specific composite models, it is under a set of valid names.

        Args:
            decoder (`Optional[bool]`, *optional*, defaults to `False`):
                If set to `True`, then only search for decoder config names.
        """
        # Overridden for deeply nested config like Qwen2-Omni. We don't have any omni model
        # except for Qwen yet. This has to be generalized if more deeply nested configs are
        # added. NOTE: currently method used only by vLLM
        return self.thinker_config.get_text_config()


class Gemma3WithTalkerPreTrainedModel(Gemma3PreTrainedModel, PreTrainedModel):
    config_class = Gemma3WithTalkerConfig
    _supports_static_cache = True

    def _init_weights(self, module):
        # important: this ported version of Qwen2.5OmniThinker isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else 0.02
        )

        if isinstance(
            module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d)
        ):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Gemma3WithTalkerAudioProjector):
            module.input_proj_weight.data.zero_()


# class Gemma3WithTalkerModel(Gemma3WithTalkerPreTrainedModel):
#     def __init__(self,config:Gemma3WithTalkerConfig):
#         super().__init__(config)
#         self.thinker = AutoModel.from_config(config=config.thinker_config)


############################
#      Start Thinker       #
############################


@dataclass
class Gemma3ThinkerCausalLMOutputWithPast(Gemma3CausalLMOutputWithPast):
    pass


# @auto_docstring(
#     custom_intro="""
#     The Gemma3Thinker model which consists of a language model.
#     """
# )
class Gemma3ThinkerForConditionalGeneration(
    Gemma3ForConditionalGeneration, GenerationMixin
):
    pass


############################
#    Start Gemma3 Talker   #
############################


@dataclass
class Gemma3TalkerCausalLMOutputWithPast(Qwen2_5OmniTalkerCausalLMOutputWithPast):
    pass


class Gemma3TalkerForConditionalGeneration(Qwen2_5OmniTalkerForConditionalGeneration):
    pass


class Gemma3Token2WavModel(Qwen2_5OmniToken2WavModel):
    pass


############################
# Start Gemma3 With Talker #
############################


class Gemma3WithTalkerAudioProjector(nn.Module):
    def __init__(self, config: Gemma3WithTalkerConfig) -> None:
        super().__init__()
        self.input_proj_weight = nn.Parameter(
            torch.zeros(
                config.thinker_config.text_config.hidden_size,
                config.talker_config.embedding_size,
            )
        )

    def forward(self, input: torch.Tensor):
        return self.input_proj_weight(input)


@auto_docstring(
    custom_intro="""
    The full Gemma3 Talker model, a multimodal model composed of 3 sub-models:
    - [`Gemma3ForConditionalGeneration`]:
    a causal auto-regressive transformer takes text, image as input and predict text tokens.
    - [`Gemma3TalkerForConditionalGeneration`]:
    a causal auto-regressive transformer takes thinker hidden states and response as input and predict speech tokens.
    - [`Gemma3Token2WavModel`]:
    a DiT model take speech tokens as input and predict mel spectrogram and a BigVGAN vocoder take mel spectrogram as input and predict waveform.
    """
)
class Gemma3WithTalkerForConditionalGeneration(
    Gemma3WithTalkerPreTrainedModel, GenerationMixin
):
    config_class = Gemma3WithTalkerConfig

    def __init__(self, config):
        super().__init__(config)
        logger.debug("Inside conditional generation and config is %s", config)

        self.thinker = Gemma3ThinkerForConditionalGeneration(config.thinker_config)

        self.has_talker = config.enable_audio_output
        self.speaker_map = {}
        # --- Initialize Projection Layers ---
        thinker_hidden_dim = config.thinker_config.text_config.hidden_size
        talker_expected_dim = config.talker_config.embedding_size

        self.projection_thinker_L0 = nn.Linear(thinker_hidden_dim, talker_expected_dim)
        self.projection_thinker_LN = nn.Linear(thinker_hidden_dim, talker_expected_dim)
        self.projection_thinker_vocab = nn.Linear(
            thinker_hidden_dim, talker_expected_dim
        )
        logger.info(
            f"Initialized projection layers: L0, LN, Vocab from {thinker_hidden_dim} to {talker_expected_dim}"
        )

        # Mark them for potential special initialization (though not strictly necessary with current _init_weights)
        # self.projection_thinker_L0._is_projection_layer = True
        # self.projection_thinker_LN._is_projection_layer = True
        # self.projection_thinker_vocab._is_projection_layer = True
        # --- End Projection Layers ---
        if config.enable_audio_output:
            self.enable_talker()
        self.post_init()

    def enable_talker(self):
        self.talker = Gemma3TalkerForConditionalGeneration(self.config.talker_config)
        self.token2wav = Gemma3Token2WavModel(self.config.token2wav_config)
        self.token2wav.float()
        self.has_talker = True

    def load_speakers(self, path):
        check_torch_load_is_safe()
        for key, value in torch.load(path, weights_only=True).items():
            self.speaker_map[key] = value
        logger.info("Speaker {} loaded".format(list(self.speaker_map.keys())))

    def disable_talker(self):
        if hasattr(self, "talker"):
            del self.talker
        if hasattr(self, "token2wav"):
            del self.token2wav
        self.has_talker = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs,
    ):
        logger_v2.info(f"Inside from pretrained")
        logger_v2.debug(f"Config {config}")
        logger_v2.debug(f"Model args")
        print(*model_args)
        logger_v2.debug(f"kwargs")
        print(**kwargs)
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        spk_path = cached_file(
            pretrained_model_name_or_path,
            "spk_dict.pt",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if spk_path is None:
            raise ValueError(
                f"""{pretrained_model_name_or_path}/{spk_path} not exists"""
            )
        model.load_speakers(spk_path)

        return model

    @torch.no_grad()
    # TODO: raushan, defaults should be saved in generation config
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        speaker: str = "Chelsie",
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        r"""
        Generate text response and audio from input.

        Args:
            input_ids (`Optional[torch.Tensor]`, *optional*):
                Input ids, should obtain from processor.
            speaker (`str` , defaults to "Chelsie"):
                Which speaker should be used in audio response.
            return_audio (`Optional[bool]`, *optional*):
                Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
            kwargs (*optional*):
                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                - With a *thinker_*, *talker_*, *token2wav_* prefix, they will be input for the `generate` method of the
                thinker, talker and token2wav respectively. It has the priority over the keywords without a prefix.
        Returns:
            When `return_audio=False`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
            When `return_audio=True`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
                - **Audio waveform** (`torch.Tensor`): Generated audio waveform.
        """
        if speaker not in self.speaker_map:
            raise ValueError(
                f"{speaker} is not available, available speakers: {self.speaker_map.keys()}"
            )
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initialized. Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker
        if input_ids.shape[0] != 1 and return_audio:
            raise NotImplementedError(
                "Gemma3Talker currently does not support batched inference with audio output"
            )

        # shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
        }
        talker_kwargs = {
            "max_new_tokens": talker_max_new_tokens,
            "do_sample": talker_do_sample,
            "top_k": talker_top_k,
            "top_p": talker_top_p,
            "temperature": talker_temperature,
            "eos_token_id": talker_eos_token_id,
            "repetition_penalty": talker_repetition_penalty,
        }
        token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            # Process special input values
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                # shared_kwargs[key] = value
                continue

        # Merge kwargs
        # for key, value in shared_kwargs.items():
        #     if key not in thinker_kwargs:
        #         thinker_kwargs[key] = value
        #     if key not in talker_kwargs:
        #         talker_kwargs[key] = value
        #     if key not in token2wav_kwargs:
        #         token2wav_kwargs[key] = value
        speaker_params = self.speaker_map[speaker]

        # 1. Generate from thinker module
        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True

        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

        if not generate_audio:
            return thinker_result

        # 2. Generate speech tokens from talker module
        embeds_to_talker = (
            thinker_result.hidden_states[0][0].clone().to(self.talker.device)
        )

        if thinker_kwargs.get("input_features", None) is not None:
            audio_ids_mask = input_ids == self.config.thinker_config.audio_token_index
            audio_mask = (
                audio_ids_mask.unsqueeze(-1)
                .expand_as(embeds_to_talker)
                .to(embeds_to_talker.device)
            )
            audio_mask_tensor = torch.zeros(
                [audio_ids_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=self.talker.device,
            )
            embeds_to_talker.masked_scatter_(audio_mask, audio_mask_tensor)
        if thinker_kwargs.get("pixel_values", None) is not None:
            image_ids_mask = input_ids == self.config.thinker_config.image_token_index
            image_mask = (
                image_ids_mask.unsqueeze(-1)
                .expand_as(embeds_to_talker)
                .to(embeds_to_talker.device)
            )
            image_mask_tensor = torch.zeros(
                [image_ids_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=self.talker.device,
            )
            embeds_to_talker.masked_scatter_(image_mask, image_mask_tensor)
        if thinker_kwargs.get("pixel_values_videos", None) is not None:
            video_ids_mask = input_ids == self.config.thinker_config.video_token_index
            video_mask = (
                video_ids_mask.unsqueeze(-1)
                .expand_as(embeds_to_talker)
                .to(embeds_to_talker.device)
            )
            video_mask_tensor = torch.zeros(
                [video_ids_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=self.talker.device,
            )
            embeds_to_talker.masked_scatter_(video_mask, video_mask_tensor)
        processed_thinker_hidden = (
            (embeds_to_talker,) + thinker_result.hidden_states[0][1:],
        ) + thinker_result.hidden_states[1:]

        thinker_token_embeds = [
            self.projection_thinker_L0(token_hidden_states[0]).to(self.talker.device)
            for token_hidden_states in processed_thinker_hidden
        ]
        thinker_hidden_states = [
            self.projection_thinker_LN(token_hidden_states[-1]).to(self.talker.device)
            for token_hidden_states in processed_thinker_hidden
        ]

        # processed_thinker_hidden = (
        #     (embeds_to_talker,) + thinker_result.hidden_states[0][1:],
        # ) + thinker_result.hidden_states[1:]

        # thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(
        #     self.talker.device
        # )
        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(
            self.talker.device
        )

        talker_text_bos_token = speaker_params["bos_token"]
        talker_input_text_ids = torch.cat(
            [
                input_ids.to(self.talker.device),
                torch.tensor(
                    [[talker_text_bos_token]],
                    dtype=torch.long,
                    device=self.talker.device,
                ),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )

        talker_input_ids = torch.cat(
            [
                torch.full_like(
                    input_ids,
                    fill_value=self.talker.codec_mask_token,
                    device=self.talker.device,
                ),
                torch.tensor(
                    [[self.talker.codec_pad_token]],
                    dtype=torch.long,
                    device=self.talker.device,
                ),
                torch.tensor(
                    [[self.talker.codec_bos_token]],
                    dtype=torch.long,
                    device=self.talker.device,
                ),
            ],
            dim=1,
        )

        thinker_embed_tokens = self.thinker.get_input_embeddings()
        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(
            thinker_token_embeds[1:], dim=1
        )
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_text_bos_token = torch.tensor(
            [[talker_text_bos_token]], dtype=torch.long, device=self.thinker.device
        )
        talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_token).to(
            self.talker.device
        )
        talker_text_bos_embed = self.projection_thinker_vocab(talker_text_bos_embed).to(
            self.talker.device
        )
        talker_inputs_embeds = torch.cat(
            [
                talker_inputs_embeds,
                talker_text_bos_embed,
                thinker_reply_part[:, :1, :],
            ],
            dim=1,
        )

        eos_embedding = thinker_embed_tokens(
            torch.tensor(
                [[self.talker.text_eos_token]],
                dtype=torch.long,
                device=self.thinker.device,
            )
        ).to(self.talker.device)
        eos_embedding = self.projection_thinker_vocab(eos_embedding).to(
            self.talker.device
        )

        pad_embedding = thinker_embed_tokens(
            torch.tensor(
                [[self.talker.text_pad_token]],
                dtype=torch.long,
                device=self.thinker.device,
            )
        ).to(self.talker.device)
        pad_embedding = self.projection_thinker_vocab(pad_embedding).to(
            self.talker.device
        )

        thinker_reply_part = torch.cat(
            [
                thinker_reply_part[:, 1:, :],
                eos_embedding,
                pad_embedding,
            ],
            dim=1,
        )

        talker_attention_mask = None
        if "attention_mask" in kwargs:
            talker_attention_mask = torch.cat(
                [kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))],
                dim=1,
            ).to(self.talker.device)

        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            suppress_tokens=[self.talker.codec_bos_token],
            **{
                k: (v.to(self.talker.device) if torch.is_tensor(v) else v)
                for k, v in talker_kwargs.items()
            },
        )
        talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]

        # 3. Generate wavs from code
        if self.token2wav.dtype != torch.float:
            self.token2wav.float()

        wav = self.token2wav(
            talker_generate_codes.to(self.token2wav.device),
            conditioning=speaker_params["cond"].to(self.token2wav.device).float(),
            reference_mel=speaker_params["ref_mel"].to(self.token2wav.device).float(),
            **token2wav_kwargs,
        )

        return thinker_result.sequences, wav.float()


__all__ = [
    "Gemma3ThinkerConfig",
    "Gemma3TalkerConfig",
    "Gemma3Token2WavConfig",
    "Gemma3WithTalkerConfig",
    "Gemma3WithTalkerPreTrainedModel",
    "Gemma3ThinkerCausalLMOutputWithPast",
    "Gemma3ThinkerForConditionalGeneration",
    "Gemma3TalkerCausalLMOutputWithPast",
    "Gemma3TalkerForConditionalGeneration",
    "Gemma3Token2WavModel",
    "Gemma3WithTalkerForConditionalGeneration",
]
