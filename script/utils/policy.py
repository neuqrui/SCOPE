"""Local chat policy used by the debug rollout. Does not touch trainer code."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalChatPolicy:
    """Generate one assistant turn from a user string, with the same chat template as rollout."""

    def __init__(
        self,
        model_path: str,
        *,
        max_prompt_length: int = 4096,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        truncation: str = "error",
        device: str | None = None,
    ):
        self.max_prompt_length = int(max_prompt_length)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.truncation = truncation
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() and device is None else None,
        )
        if device is not None and not torch.cuda.is_available():
            self.model.to(device)
        elif device is not None and torch.cuda.is_available() and getattr(self.model, "hf_device_map", None) is None:
            self.model.to(device)
        self.model.eval()

    def render_prompt(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    @torch.inference_mode()
    def generate(
        self,
        user_text: str,
        temperature: float | None = None,
        max_prompt_length: int | None = None,
    ) -> dict:
        prompt = self.render_prompt(user_text)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = int(encoded["input_ids"].shape[-1])
        limit = self.max_prompt_length if max_prompt_length is None else int(max_prompt_length)
        if prompt_len > limit:
            if self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {prompt_len} exceeds max_prompt_length={limit}. "
                    "This debug runner follows data.truncation=error and does not left-truncate."
                )
            if self.truncation == "left":
                encoded["input_ids"] = encoded["input_ids"][:, -limit:]
                encoded["attention_mask"] = encoded["attention_mask"][:, -limit:]
                prompt_len = limit
            else:
                raise ValueError(f"Unsupported truncation mode: {self.truncation}")

        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        temperature = self.temperature if temperature is None else float(temperature)
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        output = self.model.generate(**encoded, **gen_kwargs)
        new_tokens = output[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {
            "user_text": user_text,
            "rendered_prompt": prompt,
            "prompt_tokens": prompt_len,
            "response": text,
            "response_ids": new_tokens.detach().cpu().tolist(),
        }

    def _encode_user(self, user_text: str, max_prompt_length: int | None = None) -> torch.Tensor:
        prompt = self.render_prompt(user_text)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids = encoded["input_ids"][0]
        limit = self.max_prompt_length if max_prompt_length is None else int(max_prompt_length)
        if prompt_ids.numel() > limit:
            if self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {int(prompt_ids.numel())} exceeds max_prompt_length={limit}."
                )
            if self.truncation != "left":
                raise ValueError(f"Unsupported truncation mode: {self.truncation}")
            prompt_ids = prompt_ids[-limit:]
        return prompt_ids

    @torch.inference_mode()
    def score_tokens(
        self,
        user_text: str,
        response_ids: list[int],
        *,
        logit_temperature: float = 1.0,
        max_prompt_length: int | None = None,
    ) -> dict:
        """Log-prob and entropy of a fixed token sequence under this prompt.

        Logits are divided by ``logit_temperature`` before the softmax, matching
        ``dp_actor`` when it computes log-probs. Entropy is the next-token
        distribution entropy at each response position.
        """
        if not response_ids:
            return {"logprob": [], "entropy": []}
        prompt_ids = self._encode_user(user_text, max_prompt_length=max_prompt_length)
        resp = torch.tensor(response_ids, dtype=torch.long)
        context = int(getattr(self.model.config, "max_position_embeddings", 32768))
        if int(prompt_ids.numel()) + int(resp.numel()) > context:
            raise RuntimeError(
                f"prompt+response length {int(prompt_ids.numel()) + int(resp.numel())} exceeds context {context}."
            )
        device = next(self.model.parameters()).device
        input_ids = torch.cat([prompt_ids, resp]).unsqueeze(0).to(device)
        logits = self.model(input_ids=input_ids, use_cache=False).logits[0]
        prompt_len = int(prompt_ids.numel())
        resp_logits = logits[prompt_len - 1 : prompt_len - 1 + resp.numel()].float()
        temperature = float(logit_temperature)
        if temperature <= 0:
            raise ValueError("logit_temperature must be positive")
        resp_logits = resp_logits / temperature
        log_z = torch.logsumexp(resp_logits, dim=-1)
        labels = resp.to(device)
        chosen = resp_logits.gather(1, labels.unsqueeze(-1)).squeeze(-1) - log_z
        probs = torch.softmax(resp_logits, dim=-1)
        entropy = log_z - (probs * resp_logits).sum(dim=-1)
        return {
            "logprob": chosen.detach().cpu().tolist(),
            "entropy": entropy.detach().cpu().tolist(),
        }
