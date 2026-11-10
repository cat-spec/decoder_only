#最小化测试
import torch
from disttrain.model import DecoderLayer,MymodeConfig

def test_config():
    config = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                  max_seq_len=16, mlp_hidden_dim=64)
    # config.update(overrides)
    return MymodeConfig(**config)

def test_forward_shape_and_finite_values():
    torch.manual_seed(1)
    model = DecoderLayer(test_config())
    output = model(torch.randint(0, 64, (2, 8)))
    assert output.shape == (2, 8, 64)
    assert torch.isfinite(output).all()

if __name__ == '__main__':
    test_forward_shape_and_finite_values()