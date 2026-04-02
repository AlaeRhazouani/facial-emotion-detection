import torch
import pytest
from src.model import EmotionCNN, build_model


class TestEmotionCNN:

    def test_output_shape(self):
        """Output must be (batch_size, 7)"""
        model = build_model(num_classes=7)
        x = torch.randn(4, 1, 48, 48)
        output = model(x)
        assert output.shape == (4, 7), f"Expected (4, 7), got {output.shape}"

    def test_single_image(self):
        """Model must handle single image input"""
        model = build_model(num_classes=7)
        x = torch.randn(1, 1, 48, 48)
        output = model(x)
        assert output.shape == (1, 7), f"Expected (1, 7), got {output.shape}"

    def test_num_classes_respected(self):
        """build_model must respect num_classes parameter"""
        for n in [5, 7, 10]:
            model = build_model(num_classes=n)
            x = torch.randn(1, 1, 48, 48)
            output = model(x)
            assert output.shape[1] == n, f"Expected {n} classes, got {output.shape[1]}"

    def test_dropout_train_vs_eval(self):
        """Output must differ between train and eval mode due to dropout"""
        model = build_model()
        x = torch.randn(1, 1, 48, 48)

        model.train()
        out_train_1 = model(x).detach()
        out_train_2 = model(x).detach()

        model.eval()
        out_eval_1 = model(x).detach()
        out_eval_2 = model(x).detach()

        # In eval mode outputs must be identical (no dropout)
        assert torch.allclose(out_eval_1, out_eval_2), \
            "Eval mode outputs should be identical"

        # In train mode outputs should differ (dropout active)
        assert not torch.allclose(out_train_1, out_train_2), \
            "Train mode outputs should differ due to dropout"

    def test_build_model_factory(self):
        """build_model must return EmotionCNN instance"""
        model = build_model()
        assert isinstance(model, EmotionCNN), \
            f"Expected EmotionCNN, got {type(model)}"

    def test_model_has_four_blocks(self):
        """Model must have block1 through block4"""
        model = build_model()
        for block in ["block1", "block2", "block3", "block4"]:
            assert hasattr(model, block), f"Model missing {block}"

    def test_no_nan_in_output(self):
        """Output must not contain NaN values"""
        model = build_model()
        model.eval()
        x = torch.randn(4, 1, 48, 48)
        output = model(x)
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_gradients_flow(self):
        """Gradients must flow back through the model"""
        model = build_model()
        x = torch.randn(1, 1, 48, 48)
        output = model(x)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"No gradient for {name}"