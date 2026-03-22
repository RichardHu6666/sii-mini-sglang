from __future__ import annotations

from types import SimpleNamespace

import pytest

from minisgl.engine import engine as engine_mod


def _make_config(
    *,
    ep_size: int,
    moe_backend: str = "auto",
    is_moe: bool = True,
    num_experts: int = 8,
    cuda_graph_max_bs: int | None = 16,
):
    model_config = SimpleNamespace(is_moe=is_moe, num_experts=num_experts)
    return SimpleNamespace(
        attention_backend="auto",
        page_size=1,
        model_config=model_config,
        moe_backend=moe_backend,
        ep_size=ep_size,
        cuda_graph_max_bs=cuda_graph_max_bs,
    )


def test_adjust_config_auto_uses_ep_backend_when_ep_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine_mod, "is_sm100_supported", lambda: False)
    monkeypatch.setattr(engine_mod, "is_sm90_supported", lambda: False)

    cfg = _make_config(ep_size=4, moe_backend="auto", is_moe=True, num_experts=8)
    engine_mod._adjust_config(cfg)

    assert cfg.moe_backend == "ep"
    assert cfg.cuda_graph_max_bs == 0


def test_adjust_config_auto_uses_fused_when_ep_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine_mod, "is_sm100_supported", lambda: False)
    monkeypatch.setattr(engine_mod, "is_sm90_supported", lambda: False)

    cfg = _make_config(ep_size=1, moe_backend="auto", is_moe=True)
    engine_mod._adjust_config(cfg)

    assert cfg.moe_backend == "fused"


def test_adjust_config_rejects_fused_backend_when_ep_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine_mod, "is_sm100_supported", lambda: False)
    monkeypatch.setattr(engine_mod, "is_sm90_supported", lambda: False)

    cfg = _make_config(ep_size=4, moe_backend="fused", is_moe=True, num_experts=8)
    with pytest.raises(AssertionError, match="requires --moe-backend ep"):
        engine_mod._adjust_config(cfg)
