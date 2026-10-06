"""Pinned Whisper/projector/Qwen loading and completed-device stage measurements."""

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load_file
from torch import Tensor
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3_5ForCausalLM,
    WhisperFeatureExtractor,
    WhisperModel,
)
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from voice_worker.cache import continuation_mask, continuation_positions, join_caches, split_cache
from voice_worker.config import WorkerConfig
from voice_worker.detokenizer import token_bytes
from voice_worker.projector import SpeechProjector

USER_PREFIX = (248045, 846, 198)
ASSISTANT_SUFFIX = (248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271)
EOS = 248046
NEWLINE = 198


@dataclass
class StageTimer:
    device: torch.device
    started_at: float = 0.0
    ended_at: float = 0.0
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None

    def start(self) -> None:
        self.started_at = time.perf_counter()
        if self.device.type == "cuda":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record(torch.cuda.current_stream(self.device))

    def finish(self) -> None:
        if self.end_event is not None:
            self.end_event.record(torch.cuda.current_stream(self.device))
        self.ended_at = time.perf_counter()

    def elapsed_ms(self) -> float:
        if self.start_event is not None and self.end_event is not None:
            return self.start_event.elapsed_time(self.end_event)
        return (self.ended_at - self.started_at) * 1000


@dataclass(frozen=True)
class ForwardBatch:
    caches: tuple[DynamicCache, ...]
    token_ids: tuple[int, ...]
    timer: StageTimer


def pcm_waveform(audio: bytes) -> NDArray[np.float32]:
    if not audio or len(audio) % 2 or len(audio) > 960_000:
        raise ValueError("Audio must be nonempty mono PCM16/16kHz, at most thirty seconds")
    return np.frombuffer(audio, dtype="<i2").astype(np.float32) / np.float32(32768)


def pseudo_token_count(samples: int) -> int:
    return ((samples + 319) // 320 + 4) // 5


class SpeechModel:
    def __init__(self, configuration: WorkerConfig) -> None:
        self.device = torch.device(configuration.device)
        if not torch.cuda.is_available():
            raise ValueError("The production PyTorch worker requires a CUDA device")
        checkpoint = configuration.model.projector_checkpoint
        with checkpoint.open("rb") as stream:
            checksum = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
            digest = checksum.hexdigest()
        if digest != configuration.model.projector_sha256:
            raise ValueError("Projector checkpoint SHA256 differs from the deployment manifest")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            configuration.model.language_model, revision=configuration.model.language_revision
        )
        self.text_config = Qwen3_5TextConfig.from_pretrained(
            configuration.model.language_model, revision=configuration.model.language_revision
        )
        self.language_model: Qwen3_5ForCausalLM = Qwen3_5ForCausalLM.from_pretrained(
            configuration.model.language_model,
            revision=configuration.model.language_revision,
            config=self.text_config,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device)
        self.encoder: WhisperEncoder = WhisperModel.from_pretrained(
            configuration.model.encoder_model,
            revision=configuration.model.encoder_revision,
            dtype=torch.bfloat16,
        ).encoder.to(self.device)
        self.extractor = WhisperFeatureExtractor.from_pretrained(
            configuration.model.encoder_model, revision=configuration.model.encoder_revision
        )
        self.projector = SpeechProjector().to(self.device, dtype=torch.float32)
        self.projector.load_state_dict(
            load_file(str(checkpoint), device=str(self.device)), strict=True
        )
        for module in (self.language_model, self.encoder, self.projector):
            module.requires_grad_(False)
            module.eval()
        self.language_model.gradient_checkpointing_disable()
        self._validate_template()
        self.model_id = hashlib.sha256(
            configuration.model.model_dump_json().encode("utf-8")
        ).hexdigest()

    def _validate_template(self) -> None:
        prefix = self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        suffix = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        if tuple(prefix) != USER_PREFIX or tuple(suffix) != ASSISTANT_SUFFIX:
            raise ValueError("Tokenizer does not match the trained prompt markers")
        if self.tokenizer.eos_token_id != EOS or self.text_config.hidden_size != 2048:
            raise ValueError("Tokenizer EOS or language-model width differs from the checkpoint")

    def embed(self, token_ids: Sequence[int]) -> Tensor:
        indices = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        return self.language_model.get_input_embeddings()(indices)

    def token_piece(self, token_id: int) -> bytes:
        if token_id in self.tokenizer.all_special_ids:
            return b""
        return token_bytes(cast(str, self.tokenizer.convert_ids_to_tokens(token_id)))

    def encode_audio(self, audio: Sequence[bytes]) -> tuple[tuple[Tensor, ...], StageTimer]:
        waveforms = [pcm_waveform(packet) for packet in audio]
        extracted = self.extractor(
            waveforms, sampling_rate=16000, return_tensors="pt", padding="max_length"
        )
        # BatchFeature's tensor map is the only SDK dictionary boundary in the serving path.
        input_features = cast(Tensor, extracted["input_features"])
        timer = StageTimer(self.device)
        timer.start()
        features = input_features.to(self.device, dtype=torch.bfloat16)
        output: BaseModelOutput = self.encoder(features, return_dict=True)
        projected = tuple(
            self.projector(
                output.last_hidden_state[index, : min(1500, math.ceil(len(waveform) / 320))]
            ).to(torch.bfloat16)
            for index, waveform in enumerate(waveforms)
        )
        timer.finish()
        return projected, timer

    def forward_batch(
        self, embeddings: Sequence[Tensor], caches: Sequence[DynamicCache]
    ) -> ForwardBatch:
        lengths = [cache.get_seq_length() for cache in caches]
        appended = embeddings[0].shape[0]
        assert all(sequence.shape[0] == appended for sequence in embeddings)
        timer = StageTimer(self.device)
        timer.start()
        joined = join_caches(caches, self.text_config)
        output: CausalLMOutputWithPast = self.language_model(
            inputs_embeds=torch.stack(tuple(embeddings)),
            attention_mask=continuation_mask(lengths, appended, self.device),
            position_ids=continuation_positions(lengths, appended, self.device),
            past_key_values=joined,
            use_cache=True,
            logits_to_keep=1,
        )
        separated = split_cache(joined, lengths, appended, self.text_config)
        proposed = output.logits[:, -1].argmax(dim=-1)
        timer.finish()
        # Returning scalar token IDs already synchronizes this batch, so events need no extra sync.
        tokens = tuple(cast(list[int], proposed.tolist()))
        return ForwardBatch(separated, tokens, timer)

    def warmup(self) -> None:
        with torch.inference_mode():
            projected, _ = self.encode_audio((bytes(3200),))
            prompt = torch.cat(
                (self.embed(USER_PREFIX), projected[0], self.embed(ASSISTANT_SUFFIX))
            )
            first = self.forward_batch((prompt,), (DynamicCache(config=self.text_config),))
            self.forward_batch((self.embed((first.token_ids[0],)),), first.caches)
