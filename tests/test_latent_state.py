"""CPU architecture and training-contract tests; no pretrained model download."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffsynth.models.latent_state import (
    LatentStateInitializer, LatentStateTransition, LatentStateRollout,
    GameTransitionTokenizer, LatentTransitionPredictor, FiniteScalarQuantizer,
)
from diffsynth.models.stateplay_dit import StatePlayModel, StateVideoDiTBlock
from diffsynth.pipelines.stateplay import StatePlayTrainingPipeline, model_fn_stateplay
from training.latent_state_training import (
    TransitionTokenizerLoss, FlowMatchSFTLossWithLatentState,
    load_transition_tokenizer, resolve_mode_config,
)
from training.train import StatePlayTrainingModule, FlowMatchSFTLossWithStateWeighted, build_parser


def tiny_model(explicit=False, tokenizer_only=False):
    return StatePlayModel(
        dim=48, in_dim=4, ffn_dim=96, out_dim=4, text_dim=24, freq_dim=16,
        eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2,
        has_image_input=False, seperated_timestep=True, fuse_vae_embedding_in_latents=True,
        num_buttons=11, use_latent_state=not explicit and not tokenizer_only,
        use_explicit_state=explicit, use_transition_tokenizer=not explicit,
        latent_state_dim=16, latent_state_num_heads=4,
        transition_tokenizer_dim=16, transition_tokenizer_heads=4)


def tiny_pipe(explicit=False, tokenizer_only=False):
    pipe = StatePlayTrainingPipeline(device="cpu", torch_dtype=torch.float32)
    pipe.dit = tiny_model(explicit, tokenizer_only)
    pipe.scheduler.set_timesteps(10, training=True)
    pipe.units = []
    return pipe


def batch():
    clean = torch.randn(2, 4, 4, 4, 4)
    return {"input_latents": clean, "first_frame_latents": clean[:, :, :1].clone(),
            "fuse_vae_embedding_in_latents": True, "context": torch.randn(2, 6, 24),
            "keyboard_action": torch.randint(2, (2, 13, 11)).float(),
            "use_gradient_checkpointing": True}


class LatentStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        torch.manual_seed(17)

    def test_initializer_shape(self):
        with torch.no_grad():
            self.assertEqual(LatentStateInitializer()(torch.randn(2, 390, 3072)).shape, (2, 4, 256))

    def test_transition_shape(self):
        self.assertEqual(LatentStateTransition()(torch.randn(2, 4, 256), torch.randn(2, 11)).shape,
                         (2, 4, 256))

    def test_rollout_shape(self):
        with torch.no_grad():
            low, high = LatentStateRollout()(torch.randn(2, 390, 3072), torch.randn(2, 25, 11))
        self.assertEqual(low.shape, (2, 25, 4, 256))
        self.assertEqual(high.shape, (2, 100, 3072))

    def test_tokenizer_shape(self):
        with torch.no_grad():
            result = GameTransitionTokenizer()(torch.randn(2, 25 * 390, 3072), latent_frames=25)
        self.assertEqual(result["z_indices"].shape, (2, 24, 4))
        self.assertEqual(result["z_embs"].shape, (2, 24, 4, 256))
        self.assertEqual(result["loss_tok"].ndim, 0)
        self.assertEqual(result["z_indices"].dtype, torch.long)
        self.assertTrue(((result["z_indices"] >= 0) & (result["z_indices"] < 512)).all())
        self.assertFalse(result["target_delta"].requires_grad)

    def test_predictor_shape(self):
        logits = LatentTransitionPredictor()(torch.randn(2, 25, 4, 256), torch.randn(2, 25, 11))
        self.assertEqual(logits.shape, (2, 24, 4, 512))

    def test_fsq_all_codes_and_gradient(self):
        bins = torch.cartesian_prod(torch.arange(8), torch.arange(8), torch.arange(8))
        centers = (bins.float() * 2 / 7 - 1).clamp(-0.9999, 0.9999)
        u = torch.atanh(centers).requires_grad_()
        quantized, indices = FiniteScalarQuantizer()(u)
        torch.testing.assert_close(indices, torch.arange(512))
        quantized.sum().backward()
        self.assertTrue((u.grad > 0).all())

    def test_stage1_backward_and_checkpoint(self):
        pipe = tiny_pipe(tokenizer_only=True)
        pipe.requires_grad_(False)
        pipe.dit.transition_tokenizer.requires_grad_(True)
        inputs = batch()
        inputs["state"] = object()
        pipe.model_fn = lambda **kwargs: self.fail("Stage 1 must not denoise video")
        loss = TransitionTokenizerLoss(pipe, **inputs)
        loss.backward()
        for name, param in pipe.dit.named_parameters():
            if name.startswith("transition_tokenizer."):
                self.assertIsNotNone(param.grad, name)
            else:
                self.assertIsNone(param.grad, name)
        self.assertTrue(any(p.grad.abs().sum() > 0 for p in pipe.dit.transition_tokenizer.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            module = StatePlayTrainingModule.__new__(StatePlayTrainingModule)
            torch.nn.Module.__init__(module)
            module.pipe = pipe
            path = Path(directory) / "transition_tokenizer.pt"
            module.save_transition_tokenizer(path)
            restored = tiny_model().transition_tokenizer
            load_transition_tokenizer(restored, path)
            wrong_levels = dict(restored.state_dict())
            wrong_levels["quantizer.levels"] = torch.tensor([4, 8, 16])
            torch.save(wrong_levels, path)
            with self.assertRaises(ValueError):
                load_transition_tokenizer(restored, path)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, pipe.dit.transition_tokenizer.state_dict()[key])
            torch.save({"state_dict": {"pipe.dit.transition_tokenizer." + k: v
                        for k, v in restored.state_dict().items()}}, path)
            load_transition_tokenizer(restored, path)

    def test_stage2_backward_three_steps(self):
        pipe = tiny_pipe()
        pipe.dit.transition_tokenizer.requires_grad_(False)
        pipe.dit.patch_embedding.requires_grad_(False)
        pipe.dit.encode_state = lambda *args: self.fail("Explicit state encoder called")
        pipe.dit.decode_state = lambda *args: self.fail("Explicit state decoder called")
        optimizer = torch.optim.AdamW((p for p in pipe.parameters() if p.requires_grad), lr=1e-3)
        for _ in range(3):
            inputs = batch()
            inputs["state"] = object()
            loss = FlowMatchSFTLossWithLatentState(pipe, **inputs)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name in ("latent_state_rollout.initializer", "latent_state_rollout.transition",
                         "latent_transition_predictor", "blocks.0.latent_state_attention"):
                params = dict(pipe.dit.named_modules())[name].parameters()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in params), name)
            self.assertTrue(all(p.grad is None for p in pipe.dit.transition_tokenizer.parameters()))
            self.assertEqual(pipe.last_loss_components["state_loss"], 0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    def test_future_video_and_labels_cannot_change_rollout(self):
        model = tiny_model().eval()
        torch.nn.init.normal_(model.latent_state_rollout.transition.delta.weight, std=0.02)
        inputs = batch()
        inputs["latents"] = inputs.pop("input_latents")
        inputs["timestep"] = torch.tensor([500.])
        with torch.no_grad():
            first = model_fn_stateplay(model, **inputs)
            inputs["latents"][:, :, 1:] = torch.randn_like(inputs["latents"][:, :, 1:]) * 100
            second = model_fn_stateplay(model, state=torch.randn(2, 13, 5), **inputs)
            torch.testing.assert_close(first["G_seq_low"], second["G_seq_low"], rtol=0, atol=0)
            actions = torch.randn(2, 4, 11)
            h0 = torch.randn(2, 4, 48)
            low1, _ = model.latent_state_rollout(h0, actions)
            actions[:, 2:] += 20
            low2, _ = model.latent_state_rollout(h0, actions)
            torch.testing.assert_close(low1[:, :3], low2[:, :3], rtol=0, atol=0)
            self.assertFalse(torch.allclose(low1[:, 3], low2[:, 3]))

    def test_explicit_baseline(self):
        pipe = tiny_pipe(explicit=True)
        self.assertIsInstance(pipe.dit.blocks[0], StateVideoDiTBlock)
        inputs = batch()
        inputs["state"] = torch.randn(2, 13, 5)
        loss = FlowMatchSFTLossWithStateWeighted(pipe, **inputs)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in pipe.dit.state_encoder.parameters()))

    def test_mode_validation(self):
        self.assertEqual(build_parser().parse_args(["--dataset_base_path", "."]).batch_size, 4)
        with self.assertRaises(ValueError):
            resolve_mode_config("latent_state_world_model", freeze_transition_tokenizer=False)
        with self.assertRaises(NotImplementedError):
            resolve_mode_config("latent_state_joint_finetune")
        with self.assertRaises(ValueError):
            StatePlayModel(patch_size=(2, 2, 2))

    def test_mode_freezing_and_cached_batch(self):
        for mode in ("transition_tokenizer", "latent_state_world_model"):
            module = StatePlayTrainingModule.__new__(StatePlayTrainingModule)
            torch.nn.Module.__init__(module)
            module.pipe = tiny_pipe(tokenizer_only=mode == "transition_tokenizer")
            module.train_mode = mode
            module.action_dropout_prob = module.prompt_dropout_prob = 0
            module.video_loss_weight = module.tokenizer_loss_weight = 1.0
            module.transition_loss_weight = 0.1
            module.use_transition_loss = True
            module._set_train_mode_requires_grad()
            module.train()
            self.assertEqual(module.pipe.dit.transition_tokenizer.training, mode == "transition_tokenizer")
            sample = batch()
            cached = []
            for i in range(2):
                shared = {k: v[i:i+1] if torch.is_tensor(v) else v
                          for k, v in sample.items() if k != "context"}
                cached.append((shared, {"context": sample["context"][i:i+1]}, {}))
            loss = module({}, inputs=cached)
            loss.backward()
            for name, parameter in module.pipe.dit.named_parameters():
                if mode == "transition_tokenizer":
                    self.assertEqual(parameter.requires_grad, name.startswith("transition_tokenizer."))
                elif name.startswith(("transition_tokenizer.", "patch_embedding.")):
                    self.assertFalse(parameter.requires_grad)
                    self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
