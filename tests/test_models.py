"""Shape/contract tests for the model interfaces. Models are untrained; nothing here
measures recognition or translation quality."""
import pytest
import torch

from signbridge.models import SequenceFeatures, SignTranslationPipeline
from signbridge.models.reference import (CTCGlossDecoder, FrameCNNEncoder, PoseMLPEncoder, TemporalConvEncoder,
                                         TorchvisionFrameEncoder, TransformerTemporalEncoder,
                                         TransformerTranslationModel)

torch.manual_seed(0)


def test_sequence_features_validates_shapes():
    with pytest.raises(ValueError):
        SequenceFeatures(torch.zeros(2, 5, 3), torch.tensor([5]), torch.ones(2, 5, dtype=torch.bool))
    with pytest.raises(ValueError):
        SequenceFeatures(torch.zeros(1, 5, 3), torch.tensor([6]), torch.ones(1, 5, dtype=torch.bool))


def test_rgb_encoder_ignores_padding():
    enc = FrameCNNEncoder(output_dim=32).eval()
    x = torch.randn(2, 6, 3, 32, 32)
    lengths = torch.tensor([4, 6])
    out = enc(x, lengths)
    assert out.features.shape == (2, 6, 32)
    assert out.features[0, 4:].abs().sum() == 0
    x2 = x.clone()
    x2[0, 4:] = 99.0                                     # garbage in padding must not matter
    assert torch.allclose(enc(x2, lengths).features, out.features)


def test_torchvision_encoder_contract():
    enc = TorchvisionFrameEncoder("mobilenet_v2", weights=None).eval()
    out = enc(torch.randn(1, 2, 3, 64, 64), torch.tensor([2]))
    assert out.features.shape == (1, 2, enc.output_dim)


def test_pose_encoder_contract():
    enc = PoseMLPEncoder(num_keypoints=42, channels=4, output_dim=16)
    out = enc(torch.randn(2, 7, 42, 4), torch.tensor([7, 3]))
    assert out.features.shape == (2, 7, 16) and out.features[1, 3:].abs().sum() == 0


@pytest.mark.parametrize("down,T,lengths,expected", [(1, 9, [9, 5], [9, 5]), (2, 9, [9, 5], [5, 3]),
                                                     (4, 9, [9, 1], [3, 1])])
def test_temporal_conv_downsampling_updates_lengths(down, T, lengths, expected):
    enc = TemporalConvEncoder(8, 16, downsample=down)
    x = SequenceFeatures(torch.randn(2, T, 8), torch.tensor(lengths),
                         SequenceFeatures.mask_from_lengths(torch.tensor(lengths), T))
    out = enc(x)
    assert out.lengths.tolist() == expected and out.features.shape[2] == 16
    assert out.mask.sum(1).tolist() == expected


def test_transformer_temporal_encoder():
    enc = TransformerTemporalEncoder(8, d_model=16, nhead=2, layers=1).eval()
    x = SequenceFeatures(torch.randn(2, 5, 8), torch.tensor([5, 2]),
                         SequenceFeatures.mask_from_lengths(torch.tensor([5, 2]), 5))
    out = enc(x)
    assert out.features.shape == (2, 5, 16) and out.features[1, 2:].abs().sum() == 0


def test_ctc_decoder_loss_and_greedy_decode():
    dec = CTCGlossDecoder(16, vocab_size=6)
    x = SequenceFeatures(torch.randn(2, 10, 16), torch.tensor([10, 7]),
                         SequenceFeatures.mask_from_lengths(torch.tensor([10, 7]), 10))
    out = dec(x)
    assert out.log_probs.shape == (10, 2, 6)
    loss = dec.loss(out, torch.tensor([[3, 4, 5], [3, 3, 0]]), torch.tensor([3, 2]))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert dec.head.weight.grad is not None


def test_ctc_greedy_collapse_rule():
    from signbridge.models import GlossOutput
    dec = CTCGlossDecoder(4, vocab_size=4)
    path = torch.tensor([0, 2, 2, 0, 2, 3, 3, 0])            # blank=0 -> [2, 2, 3]
    lp = torch.full((8, 1, 4), -10.0)
    lp[torch.arange(8), 0, path] = 0.0
    assert dec.decode(GlossOutput(lp, torch.tensor([8]), 0)) == [[2, 2, 3]]


def test_translation_model_teacher_forcing_and_generate():
    tr = TransformerTranslationModel(memory_dim=16, vocab_size=20, d_model=16, nhead=2, layers=1).eval()
    mem = SequenceFeatures(torch.randn(2, 6, 16), torch.tensor([6, 3]),
                           SequenceFeatures.mask_from_lengths(torch.tensor([6, 3]), 6))
    tgt = torch.tensor([[2, 5, 6, 3], [2, 7, 3, 0]])
    logits = tr(mem, tgt)
    assert logits.shape == (2, 4, 20)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 20), tgt[:, 1:].reshape(-1), ignore_index=0)
    assert torch.isfinite(loss)
    gen = tr.generate(mem, max_len=5)
    assert len(gen) == 2 and all(len(g) <= 5 for g in gen)


def test_pipeline_gloss_free_and_sign2gloss2text_modes():
    vis = FrameCNNEncoder(output_dim=16)
    tmp = TemporalConvEncoder(16, 16, downsample=2)
    x, lengths = torch.randn(2, 8, 3, 32, 32), torch.tensor([8, 5])
    gloss_free = SignTranslationPipeline(vis, tmp, translator=TransformerTranslationModel(16, 12, d_model=16, nhead=2, layers=1))
    out = gloss_free(x, lengths, target_in=torch.tensor([[2, 4, 3], [2, 5, 3]]))
    assert "gloss" not in out and out["translation_logits"].shape == (2, 3, 12)
    assert out["encoder"].lengths.tolist() == [4, 3]
    full = SignTranslationPipeline(vis, tmp, gloss_decoder=CTCGlossDecoder(16, 5))
    assert full(x, lengths)["gloss"].log_probs.shape == (4, 2, 5)
    with pytest.raises(ValueError):
        SignTranslationPipeline(vis, TemporalConvEncoder(32, 16))
