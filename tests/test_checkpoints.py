"""A reloaded checkpoint must be the *same function*, not a similar one.

The figures are replayed from these files, so the property that matters is
pointwise identity of the field -- and, because the residual is formed from
autograd derivatives of the network, identity of its derivatives too. Both are
tested here, along with the failure modes that would otherwise produce a
plausible but wrong figure: a config that no longer matches the weights, and a
Fourier model whose frozen projection was redrawn instead of restored.
"""

import pytest
import torch

from pinn import derivatives, load_model, save_model
from pinn.checkpoints import FORMAT
from pinn.features import FourierMLP
from pinn.model import MLP, set_seed


def _coords(n=64, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 2, generator=g) * 2 - 1


def test_mlp_round_trip_is_bit_for_bit(tmp_path):
    set_seed(0)
    model = MLP(in_dim=2, out_dim=1, width=16, depth=3, activation="tanh")
    cfg = dict(in_dim=2, out_dim=1, width=16, depth=3, activation="tanh")
    path = save_model(model, str(tmp_path / "m.pt"), cfg, meta={"steps": 10})

    loaded, meta = load_model(path)
    x = _coords()
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)
    assert meta == {"steps": 10}


def test_round_trip_preserves_the_autograd_derivatives(tmp_path):
    """eval() must not cost the loaded model its input gradients: the residual
    figures differentiate through it."""
    set_seed(1)
    model = MLP(width=16, depth=3)
    path = save_model(model, str(tmp_path / "m.pt"),
                      dict(in_dim=2, out_dim=1, width=16, depth=3))
    loaded, _ = load_model(path)

    x = _coords(32).requires_grad_(True)
    u_ref = model(x)
    u_new = loaded(x)
    torch.testing.assert_close(
        derivatives.u_xx(u_new, x), derivatives.u_xx(u_ref, x), rtol=0, atol=0
    )


def test_fourier_projection_is_restored_not_redrawn(tmp_path):
    """B is a buffer, and the whole model is wrong if it comes back different.

    A FourierMLP rebuilt with a different feature_seed still loads a matching
    state_dict shape-wise, so this is the silent-failure case the round trip
    has to rule out.
    """
    set_seed(2)
    cfg = dict(in_dim=2, out_dim=1, width=16, depth=2,
               n_features=8, sigma=(5.0, 1.0), feature_seed=7)
    model = FourierMLP(**cfg)
    path = save_model(model, str(tmp_path / "f.pt"), cfg)
    loaded, _ = load_model(path)

    torch.testing.assert_close(loaded.features.B, model.features.B, rtol=0, atol=0)
    x = _coords()
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)

    # and a differently-seeded rebuild really is a different function, i.e. the
    # assertion above is not vacuous
    other = FourierMLP(**{**cfg, "feature_seed": 8})
    assert not torch.allclose(other.features.B, model.features.B)


def test_config_that_disagrees_with_the_weights_raises(tmp_path):
    set_seed(0)
    model = MLP(width=16, depth=3)
    path = save_model(model, str(tmp_path / "m.pt"),
                      dict(in_dim=2, out_dim=1, width=32, depth=3))  # wrong width
    with pytest.raises(RuntimeError):
        load_model(path)


def test_unknown_type_and_format_are_rejected(tmp_path):
    class NotCheckpointable(torch.nn.Module):
        pass

    with pytest.raises(ValueError):
        save_model(NotCheckpointable(), str(tmp_path / "x.pt"), {})

    path = tmp_path / "bad.pt"
    torch.save({"format": FORMAT + 99, "kind": "MLP", "config": {},
                "meta": {}, "state_dict": {}}, path)
    with pytest.raises(ValueError):
        load_model(str(path))
