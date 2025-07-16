from sentence_transformers import SentenceTransformer, models, InputExample, losses, LoggingHandler

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from torch.utils.data import DataLoader
import pickle
import json
from sentence_transformers.evaluation import BinaryClassificationEvaluator, InformationRetrievalEvaluator
from datasets import Dataset
import logging
import argparse
import os
#### Just some code to print debug information to stdout
from huggingface_hub import login
import wandb
import random
from dotenv import load_dotenv
import utils

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
    parser.add_argument("--pair_eval_path", type=str, default="", help="path to saved pair eval data")
    parser.add_argument("--data_path", type=str, default="", help="path to data")
    parser.add_argument("--round", default=1, type=str, help="training round ")
    parser.add_argument("--eval_size", default=0.2, type=float, help="number of eval data")
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")
    parser.add_argument("--wseg", action="store_true", help="word segmentation")
    parser.add_argument("--saved_model", default="saved_model", type=str, help="path to savd model directory.")
    parser.add_argument("--hub_model_id", default="", type=str, help="path to save model huggingface.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for training")
    parser.add_argument("--step", type=int, default=100, help="learning rate for training")
    parser.add_argument("--zalo", action="store_true", help="use Zalo data")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    if round == 1:
        print(f"Training round 1")
        word_embedding_model = models.Transformer(args.pretrained_model, max_seq_length=args.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print(model)
    else:
        model = SentenceTransformer(args.pretrained_model)
    save_pairs = load_pair_data(args.pair_data_path)
    print(f"There are {len(save_pairs)} pair sentences.")
    train_examples = {"question": [], "document": [], "label": []}
    eval_examples = {"question": [], "document": [], "label": []}
    loss = ContrastiveLoss(model)

    output_path = args.saved_model
    os.makedirs(output_path, exist_ok=True)
    if args.eval_size <= 0.0:
        for idx, pair in enumerate(save_pairs):
            relevant = float(pair["relevant"])
            qid = pair["qid"]
            question = pair["question"]
            document = pair["document"]
            if args.wseg:
                question = utils.word_segmentation(question)
                document = utils.word_segmentation(document)
            train_examples["question"].append(question)
            train_examples["document"].append(document)
            train_examples["label"].append(relevant)
        train_dataset = Dataset.from_dict(train_examples)
        loss = ContrastiveLoss(model)
        hub_model_id = args.hub_model_id
    
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="ALQAC-2025", name=hub_model_id+"_run")
        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir=output_path,
            # Optional training parameters:
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size*2,
            gradient_accumulation_steps=10,
            gradient_checkpointing=True,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            # batch_sampler=BatchSamplers.GROUP_BY_LABEL,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            # Optional tracking/debugging parameters:
            save_strategy="steps",
            save_steps=args.step*4,
            logging_steps=args.step,
            prompts={"query": model.prompts["query"]} if "Qwen" in args.pretrained_model else None,
            run_name=args.pretrained_model+"_run",  # Will be used in W&B if `wandb` is installed
            push_to_hub=True if hub_model_id else False,
            hub_model_id=hub_model_id if hub_model_id else None,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=loss,
        )
        trainer.train()
        wandb.finish()
        logging.info("Training finished!")
        
        print(f"Model saved to {output_path}")
    else:
        if not args.zalo:
            with open(os.path.join(args.data_path, "queries.json"), "r") as f:
                queries = json.load(f)
            qid_list = list(queries.keys())
            random.seed(42)
            random.shuffle(qid_list)
            num_eval = int(len(qid_list) * args.eval_size)
            eval_qid = qid_list[:num_eval]
            for idx, pair in enumerate(save_pairs):
                relevant = float(pair["relevant"])
                qid = pair["qid"]
                question = pair["question"]
                document = pair["document"]
                if args.wseg:
                    question = utils.word_segmentation(question)
                    document = utils.word_segmentation(document)
                if qid not in eval_qid:
                    train_examples["question"].append(question)
                    train_examples["document"].append(document)
                    train_examples["label"].append(relevant)
                else:
                    eval_examples["question"].append(question)
                    eval_examples["document"].append(document)
                    eval_examples["label"].append(relevant)
        else:
            save_eval_pairs = load_pair_data(args.pair_eval_path)
            for idx, pair in enumerate(save_pairs):
                relevant = float(pair["relevant"])
                qid = pair["qid"]
                question = pair["question"]
                document = pair["document"]
                if args.wseg:
                    question = utils.word_segmentation(question)
                    document = utils.word_segmentation(document)
                train_examples["question"].append(question)
                train_examples["document"].append(document)
                train_examples["label"].append(relevant)
                
            for idx, pair in enumerate(save_eval_pairs):
                relevant = float(pair["relevant"])
                qid = pair["qid"]
                question = pair["question"]
                document = pair["document"]
                if args.wseg:
                    question = utils.word_segmentation(question)
                    document = utils.word_segmentation(document)
                eval_examples["question"].append(question)
                eval_examples["document"].append(document)
                eval_examples["label"].append(relevant)
        print("Number of sample: ", len(train_examples["question"]))
        print("Number of eval sample: ", len(eval_examples["question"]))

        train_dataset = Dataset.from_dict(train_examples)
        eval_dataset = Dataset.from_dict(eval_examples)
        with open(os.path.join(args.data_path, "corpus.json"), "r") as f:
            corpus = json.load(f)
        with open(os.path.join(args.data_path, "queries.json"), "r") as f:
            queries = json.load(f)
        with open(os.path.join(args.data_path, "relevant_docs.json"), "rb") as f:
            relevant_docs = pickle.load(f)
        
        

        eval_questions = list(eval_dataset["question"])
        eval_documents = list(eval_dataset["document"])
        eval_labels = list(eval_dataset["label"])

        evaluator = BinaryClassificationEvaluator(
            sentences1=eval_questions,
            sentences2=eval_documents,
            labels=eval_labels,
        )

        hub_model_id = args.hub_model_id
        
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="ALQAC-2025", name=hub_model_id+"_run")
        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir=output_path,
            # Optional training parameters:
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size*2,
            gradient_accumulation_steps=10,
            gradient_checkpointing=True,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            # batch_sampler=BatchSamplers.GROUP_BY_LABEL,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            # Optional tracking/debugging parameters:
            eval_strategy="steps",
            eval_steps=args.step*2,
            save_strategy="steps",
            save_steps=args.step*4,
            logging_steps=args.step,
            load_best_model_at_end=True,
            prompts={"query": model.prompts["query"]} if "Qwen" in args.pretrained_model else None,
            run_name=args.pretrained_model+"_run",  # Will be used in W&B if `wandb` is installed
            push_to_hub=True if hub_model_id else False,
            hub_model_id=hub_model_id if hub_model_id else None,
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
        wandb.finish()
        logging.info("Training finished!")
        
        print(f"Model saved to {output_path}")

