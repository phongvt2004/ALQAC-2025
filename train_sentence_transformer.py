from sentence_transformers import SentenceTransformer, models, InputExample, losses, LoggingHandler

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from sentence_transformers.losses import ContrastiveLoss
from sentence_transformers.training_args import BatchSamplers
from torch.utils.data import DataLoader
import pickle
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from datasets import Dataset
import logging
import argparse
import os
#### Just some code to print debug information to stdout
from huggingface_hub import login
import wandb

from dotenv import load_dotenv

load_dotenv()
def load_pair_data(pair_data_path):
    with open(pair_data_path, "rb") as pair_file:
        pairs = pickle.load(pair_file)
    return pairs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", default="", type=str, help="path to your language model")
    parser.add_argument("--max_seq_length", default=256, type=int, help="maximum sequence length")
    parser.add_argument("--pair_data_path", type=str, default="", help="path to saved pair data")
    parser.add_argument("--round", default=1, type=str, help="training round ")
    parser.add_argument("--eval_size", default=0.2, type=float, help="number of eval data")
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")
    parser.add_argument("--saved_model", default="saved_model", type=str, help="path to savd model directory.")
    parser.add_argument("--hub_model_id", default="", type=str, help="path to save model huggingface.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for training")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    # if round == 1:
    #     print(f"Training round 1")
    #     word_embedding_model = models.Transformer(args.pretrained_model, max_seq_length=args.max_seq_length)
    #     pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    #     model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    #     print(model)
    # else:
    #     print("Training round 2")
    model = SentenceTransformer(args.pretrained_model)
    print(model)

    save_pairs = load_pair_data(args.pair_data_path)
    print(f"There are {len(save_pairs)} pair sentences.")
    train_examples = {"question": [], "document": [], "label": []}
    sent1 = []
    sent2 = []
    scores = []

    for idx, pair in enumerate(save_pairs):
        relevant = float(pair["relevant"])
        question = pair["question"]
        document = pair["document"]
        train_examples["question"].append(question)
        train_examples["document"].append(document)
        train_examples["label"].append(relevant)

    print("Number of sample for training: ", len(train_examples))

    dataset = Dataset.from_dict(train_examples).train_test_split(test_size=args.eval_size, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    loss = ContrastiveLoss(model)

    output_path = args.saved_model
    os.makedirs(output_path, exist_ok=True)

    evaluator = BinaryClassificationEvaluator(
        sentences1=eval_dataset["question"],
        sentences2=eval_dataset["document"],
        labels=eval_dataset["label"],
    )
    hub_model_id = args.hub_model_id
    
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project="ALQAC-2025")
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_path,
        # Optional training parameters:
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size*2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # batch_sampler=BatchSamplers.GROUP_BY_LABEL,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=2000,
        logging_steps=250,
        run_name=args.pretrained_model+"_run",  # Will be used in W&B if `wandb` is installed
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    
    if hub_model_id:
        model.push_to_hub(hub_model_id, final_output_dir)
        logger.info("Final model pushed to hub")
    
    wandb.finish()
    writer.close() # Close TensorBoard writer
    logger.info("Training finished!")
    
    print(f"Model saved to {output_path}")

