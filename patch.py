import types
from transformers.trainer import *


def patch_trainer_optimizer(
    trainer,
    lr_thinking_residual_gate=1e-4,
    thinking_residual_Lambda=1e-3,
    lr_time_conditioning=None,
):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            _special = ("thinking_residual", "adaln_", "time_progress_predictor", "sinusoidal_time_embedding")

            def _is_special(n):
                return any(s in n for s in _special)

            def _collect_group_params(predicate, use_decay):
                return [
                    p
                    for n, p in opt_model.named_parameters()
                    if predicate(n)
                    and (n in decay_parameters) is use_decay
                    and p.requires_grad
                ]

            def _append_split_groups(groups, predicate, lr):
                if lr is None:
                    return
                decay_group = _collect_group_params(predicate, use_decay=True)
                no_decay_group = _collect_group_params(predicate, use_decay=False)
                if decay_group:
                    groups.append(
                        {
                            "params": decay_group,
                            "lr": lr,
                            "weight_decay": self.args.weight_decay,
                        }
                    )
                if no_decay_group:
                    groups.append(
                        {
                            "params": no_decay_group,
                            "lr": lr,
                            "weight_decay": 0.0,
                        }
                    )

            optimizer_grouped_parameters = [
                {
                    "params": _collect_group_params(lambda n: not _is_special(n), use_decay=True),
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": _collect_group_params(lambda n: not _is_special(n), use_decay=False),
                    "lr": self.args.learning_rate,
                    "weight_decay": 0.0,
                },
            ]
            optimizer_grouped_parameters = [
                group for group in optimizer_grouped_parameters if group["params"]
            ]

            _append_split_groups(
                optimizer_grouped_parameters,
                lambda n: "thinking_residual_gate" in n,
                lr_thinking_residual_gate,
            )
            _append_split_groups(
                optimizer_grouped_parameters,
                lambda n: "thinking_residual_Lambda" in n,
                thinking_residual_Lambda,
            )

            if lr_time_conditioning is not None:
                _tc_names = ("sinusoidal_time_embedding", "adaln_proj", "time_progress_predictor")
                _append_split_groups(
                    optimizer_grouped_parameters,
                    lambda n: any(tc in n for tc in _tc_names),
                    lr_time_conditioning,
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
