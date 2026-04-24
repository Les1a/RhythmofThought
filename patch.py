"""Repository-specific trainer patching helpers for GRPO-family modes."""

import types
from transformers.trainer import *

from time_conditioning import TIME_CONDITIONING_MODULE_NAMES


def patch_trainer_optimizer(
    trainer,
    lr_thinking_residual_gate=1e-4,
    thinking_residual_Lambda=1e-3,
    lr_time_conditioning=None,
):
    """Replace the trainer optimizer builder with mode-aware parameter grouping."""
    def create_optimizer(self):
        """
        Build optimizer groups for base, residual, and time-conditioning modules.

        Each enabled parameter family is split into decay and no-decay groups.
        This keeps the repository's custom learning-rate layout close to the
        training mode instead of spreading that logic across every task script.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_groups = {
                ("base", True): [],
                ("base", False): [],
                ("thinking_residual_gate", True): [],
                ("thinking_residual_gate", False): [],
                ("thinking_residual_Lambda", True): [],
                ("thinking_residual_Lambda", False): [],
                ("time_conditioning", True): [],
                ("time_conditioning", False): [],
            }

            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue

                use_decay = name in decay_parameters
                target_group = None
                if "thinking_residual_gate" in name:
                    target_group = "thinking_residual_gate" if lr_thinking_residual_gate is not None else None
                elif "thinking_residual_Lambda" in name:
                    target_group = "thinking_residual_Lambda" if thinking_residual_Lambda is not None else None
                elif any(module_name in name for module_name in TIME_CONDITIONING_MODULE_NAMES):
                    target_group = "time_conditioning" if lr_time_conditioning is not None else None
                elif "thinking_residual" not in name:
                    target_group = "base"

                if target_group is not None:
                    optimizer_groups[(target_group, use_decay)].append(param)

            group_lrs = {
                "base": self.args.learning_rate,
                "thinking_residual_gate": lr_thinking_residual_gate,
                "thinking_residual_Lambda": thinking_residual_Lambda,
                "time_conditioning": lr_time_conditioning,
            }
            optimizer_grouped_parameters = []
            for group_name, lr in group_lrs.items():
                if lr is None:
                    continue
                for use_decay in (True, False):
                    params = optimizer_groups[(group_name, use_decay)]
                    if not params:
                        continue
                    optimizer_grouped_parameters.append(
                        {
                            "params": params,
                            "lr": lr,
                            "weight_decay": self.args.weight_decay if use_decay else 0.0,
                        }
                    )

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    trainer._old_create_optimizer = trainer.create_optimizer
    trainer.create_optimizer = types.MethodType(create_optimizer, trainer)
