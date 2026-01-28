import sys
import logging
import traceback

from transformers.utils import logging

logger = logging.get_logger(__name__)


def attempt_train(trainer, checkpoint):
    try:
        return trainer.train(resume_from_checkpoint=checkpoint), True
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt, exiting.")
        sys.exit(1)
    except Exception as e:
        logger.warning(f"Error resuming from checkpoint {checkpoint}: {e}")
        logger.warning("Full Traceback:")
        logger.warning(traceback.format_exc())
        return None, False


def find_valid_checkpoint(trainer, training_args):
    checkpoint_dirs = trainer._sorted_checkpoints(
        use_mtime=False, output_dir=training_args.output_dir
    )
    for checkpoint in reversed(
        checkpoint_dirs[1:]
    ):  # Skip the most recent as it's already attempted
        logger.info(f"Attempting to resume from {checkpoint}")
        train_result, loaded = attempt_train(trainer, checkpoint)
        if loaded:
            return train_result
    logger.info("No valid checkpoint found, starting from scratch.")
    return trainer.train(resume_from_checkpoint=None)


def train_model(trainer, training_args, data_args, train_dataset, last_checkpoint=None):
    checkpoint = (
        training_args.resume_from_checkpoint or last_checkpoint
    )  # Simplified conditional assignment
    train_result, loaded = attempt_train(trainer, checkpoint)

    if not loaded:
        train_result = find_valid_checkpoint(trainer, training_args)

    metrics = train_result.metrics
    max_train_samples = data_args.max_train_samples or len(
        train_dataset
    )  # Use or for default value
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.save_model()  # Save the model and tokenizer
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
