import logging
import shutil

from transformers.utils import logging

logger = logging.get_logger(__name__)


def delete_checkpoint(checkpoint_dir):
    try:
        logger.info(f"Deleting checkpoint {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)
        logger.info("Deletion successful.")
    except FileNotFoundError:
        logger.warning(f"Checkpoint {checkpoint_dir} does not exist.")


def cleanup_checkpoints(trainer, training_args):
    logger.info("Cleaning up checkpoints!")
    output_dir = training_args.output_dir
    checkpoints_sorted = trainer._sorted_checkpoints(
        use_mtime=False, output_dir=output_dir
    )
    logger.info(f"Current checkpoints: {checkpoints_sorted}")

    if training_args.keep_checkpoints == "none":
        # Delete the entire output directory
        try:
            logger.info(f"Deleting all checkpoints in {output_dir}")
            shutil.rmtree(output_dir)
            logger.info("Deletion successful.")
        except FileNotFoundError:
            logger.warning(f"Directory {output_dir} does not exist.")
    elif training_args.keep_checkpoints == "eval":
        for checkpoint in checkpoints_sorted:
            delete_checkpoint(checkpoint)

    logger.info("Successful completion.")
