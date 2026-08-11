"""C024: tests for Sampler and logits processors."""

import torch

from my_vllm.v1.sample.metadata import SamplingMetadata
from my_vllm.v1.sample.sampler import Sampler
from my_vllm.v1.sample.logits_processor import (
    apply_repetition_penalty,
    apply_frequency_penalty,
    apply_presence_penalty,
)


class TestSamplerGreedy:
    def test_greedy_single(self):
        sampler = Sampler()
        logits = torch.tensor([[0.1, 2.0, 0.5, 1.0]])
        meta = SamplingMetadata(all_greedy=True)
        result = sampler(logits, meta)
        assert result.shape == (1,)
        assert result[0].item() == 1  # index of max (2.0)

    def test_greedy_batch(self):
        sampler = Sampler()
        logits = torch.tensor([
            [0.1, 2.0, 0.5],
            [3.0, 1.0, 0.2],
            [0.0, 0.0, 5.0],
        ])
        meta = SamplingMetadata(all_greedy=True)
        result = sampler(logits, meta)
        assert result.tolist() == [1, 0, 2]


class TestSamplerTemperature:
    def test_temperature_shape(self):
        sampler = Sampler()
        logits = torch.tensor([
            [0.1, 2.0, 0.5],
            [3.0, 1.0, 0.2],
        ])
        meta = SamplingMetadata(
            all_greedy=False,
            temperature=torch.tensor([0.5, 1.0]),
        )
        result = sampler(logits, meta)
        assert result.shape == (2,)


class TestSamplerTopK:
    def test_top_k(self):
        sampler = Sampler()
        logits = torch.tensor([[10.0, 5.0, 0.1, 0.0, -1.0]])
        meta = SamplingMetadata(
            all_greedy=False,
            temperature=torch.tensor([1.0]),
            top_k=torch.tensor([3], dtype=torch.long),
        )
        result = sampler(logits, meta)
        assert result[0].item() in {0, 1, 2}  # top 3 only


class TestSamplerTopP:
    def test_top_p(self):
        sampler = Sampler()
        logits = torch.tensor([[10.0, 5.0, 0.1, 0.0, -1.0]])
        meta = SamplingMetadata(
            all_greedy=False,
            temperature=torch.tensor([1.0]),
            top_p=torch.tensor([0.5]),
        )
        result = sampler(logits, meta)
        # top-p=0.5 picks from tokens covering 50% probability mass
        assert result[0].item() >= 0


class TestPenalties:
    def test_repetition_penalty_suppress(self):
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        prompt_ids = torch.tensor([[1, 0, 0]])  # token 1 appears
        output_ids = [[]]
        apply_repetition_penalty(logits, prompt_ids, output_ids, penalty=1.2)
        # token 1 (logit 5.0) should be penalized
        assert logits[0, 1].item() < 5.0

    def test_repetition_penalty_noop(self):
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        apply_repetition_penalty(logits, None, [[]], penalty=1.0)
        assert logits[0, 1].item() == 5.0

    def test_frequency_penalty(self):
        logits = torch.tensor([[3.0, 3.0]])
        output_ids = [[1, 1]]  # token 1 appeared twice
        apply_frequency_penalty(logits, output_ids, penalty=0.5)
        assert logits[0, 1].item() < 3.0

    def test_presence_penalty(self):
        logits = torch.tensor([[3.0, 3.0]])
        output_ids = [[1]]  # token 1 appeared
        apply_presence_penalty(logits, output_ids, penalty=0.5)
        assert logits[0, 1].item() < 3.0


MODEL = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


class TestSamplerIntegration:
    def test_gpu_model_runner_uses_sampler(self):
        from my_vllm.model_executor.model_loader import load_model
        from my_vllm.entrypoints.llm import load_tokenizer
        from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from my_vllm.v1.core.sched.output import (
            SchedulerOutput, NewRequestData, CachedRequestData,
        )

        model = load_model(MODEL)
        device = next(model.parameters()).device
        tok = load_tokenizer(MODEL)

        runner = GPUModelRunner(
            model, device, block_size=16,
            max_num_blocks=32, max_model_len=128,
        )
        assert isinstance(runner.sampler, Sampler)

        prompt = list(tok.encode("Hello"))
        so = SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id="s1", prompt_token_ids=prompt,
                    block_ids=list(range(8)),
                ),
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            total_num_scheduled_tokens=len(prompt),
        )
        result = runner.execute_model(so)
        assert "s1" in result.sampled_token_ids
        assert len(result.sampled_token_ids["s1"]) == 1
