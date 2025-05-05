# import openai
import os
import argparse
import time
import logging
import traceback
# from dotenv import load_dotenv, find_dotenv
# import openpyxl
from pathlib import Path
import json
import re
import statistics
from typing import Dict, Any, List, Union
import torch
import csv

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core import SimpleDirectoryReader, Settings
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer, LoggingHandler
from transformers import get_linear_schedule_with_warmup
from sentence_transformers.losses import CosineSimilarityLoss, ContrastiveLoss, MultipleNegativesRankingLoss

from llama_index.core.schema import TextNode
from tqdm.notebook import tqdm
import pandas as pd
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

from llama_index.embeddings.adapter import LinearLayer
from my_adapter import (
    TwoLayerNN, 
    SelfAttentionAdapter,
    config_TwoLayerNN, 
    config_LinearLayer,
    config_SelfAttentionAdapter, 
)
from llama_index.core.embeddings import resolve_embed_model
from my_hack import MySentenceTransformersFinetuneEngine, MyEmbeddingAdapterFinetuneEngine, MySentenceTransformerWrapper
from transformers.integrations import WandbCallback


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.addHandler(LoggingHandler())
logger.setLevel(logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
def write_tocsv(file_path, row_data):
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(row_data)


def evaluate(dataset, embed_model, top_k=5, verbose=0):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=False)
    retriever = index.as_retriever(similarity_top_k=top_k)
    eval_results = []
    hit_count = 0
    for query_id, query_text in tqdm(queries.items(), desc="Evaluating", disable=True):
        retrieved_nodes = retriever.retrieve(query_text)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids
        if is_hit:
            hit_count += 1
        eval_result = {
            "query_id": query_id,
            "query_text": query_text,
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
        }
        eval_results.append(eval_result)
    hit_rate = hit_count / len(queries)
    return eval_results, hit_rate


def evaluate_st(dataset,  name, model=None, model_id=None, output_path=None):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs
    if model is None:  # finetuned base model
        model = SentenceTransformer(model_id)
        evaluator = InformationRetrievalEvaluator(
        queries,
        corpus,
        relevant_docs,
        name=name,
        show_progress_bar=False
    )
    else:  # finetuned adapter
        base_model = SentenceTransformer(model_id)
        # use the base model score functions
        score_functions = {base_model.similarity_fn_name: base_model.similarity}
        evaluator = InformationRetrievalEvaluator(
        queries,
        corpus,
        relevant_docs,
        name=name,
        show_progress_bar=False,
        score_functions=score_functions,
    )
    if output_path is not None:
        return evaluator(model, output_path=output_path)
    else:
        return evaluator(model)


def main():
    # arguments
    parser = argparse.ArgumentParser(description="BGE Embeddings Finetune and Evaluate")
    parser.add_argument("--dataset_version", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_version", type=str, required=True)
    parser.add_argument("--adapter_model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--loss_func", type=str, default=None)
    parser.add_argument("--modes", nargs='+', required=True)
    parser.add_argument("--result_output_path", type=str, required=True)
    args = parser.parse_args()

    train_data_path = os.path.join(args.dataset_path, f"train_dataset_{args.dataset_version}.json")
    eval_data_path = os.path.join(args.dataset_path, f"eval_dataset_{args.dataset_version}.json")
    base_model_dict = {
        "small-1.5": "BAAI/bge-small-en-v1.5",
        "base-1.5": "BAAI/bge-base-en-v1.5",
        "large-1.5": "BAAI/bge-large-en-v1.5"
    }
    if args.model_version not in base_model_dict:
        raise ValueError("Unsupported model_version. Must be one of {small, base, large}.")
    base_model_id = base_model_dict[args.model_version]
    
    adapter_model_dict = {
        "twolayernn": TwoLayerNN,
        "linear": LinearLayer,
        "selfattention": SelfAttentionAdapter,
    }
    
    adapter_config_dict = {
        "twolayernn": config_TwoLayerNN,
        "linear": config_LinearLayer,
        "selfattention": config_SelfAttentionAdapter,
    }
    
    loss_func_dict = {
        "cosine": CosineSimilarityLoss,
        "contrastive": ContrastiveLoss,
    }
    
    if args.loss_func not in ["cosine", "contrastive", None]:
        raise ValueError("Unsupported loss_func. Must be one of {consine, contrastive, None}.")
    if args.loss_func is not None and args.adapter_model is not None:
        raise ValueError("For training adapter, can only use the default multiple ranking loss.")
    
    if not os.path.exists(args.result_output_path):
        os.makedirs(args.result_output_path, exist_ok=True)
        
    model_output_name = "model"
    if args.adapter_model is not None:
        model_output_name += f"_{args.adapter_model}"
    if args.loss_func is not None:
        model_output_name += f"_{args.loss_func}"
    model_output_name += f"_{args.dataset_version}_{args.model_version}_e{args.epochs}_b{args.batch_size}"
    model_output_path = os.path.join(args.result_output_path, model_output_name)
    
    # load datasets
    logger.info(f"Loading dataset from {train_data_path} and {eval_data_path}")
    try:
        train_dataset = EmbeddingQAFinetuneDataset.from_json(train_data_path)
        eval_dataset = EmbeddingQAFinetuneDataset.from_json(eval_data_path)
    except FileNotFoundError as e:
        logger.error(f"Failed to load dataset. File not found: {e}")
        logger.debug(traceback.format_exc())
        return
    except Exception as e:
        logger.error(f"Failed to load dataset. Error: {e}")
        logger.debug(traceback.format_exc())
        return

    if "train" in args.modes:
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path, exist_ok=True)

        logger.info("Start training...")
        try:
            # trainer
            start_time = time.time()
            # finetune the model itself
            if args.adapter_model is None:
                finetune_engine = MySentenceTransformersFinetuneEngine(
                    dataset=train_dataset,
                    model_id=base_model_id,
                    model_output_path=model_output_path,
                    val_dataset=eval_dataset,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    loss=loss_func_dict[args.loss_func] if args.loss_func is not None else None,  # by default, MultipleNegativesRankingLoss
                )
                adapter_model=None
            # finetune with an adapter
            else:
                base_embed_model = resolve_embed_model("local:" + base_model_id)
                logger.info(base_embed_model)
                adapter_model = adapter_model_dict[args.adapter_model]
                adapter_config = adapter_config_dict[args.adapter_model][args.model_version]
                finetune_engine = MyEmbeddingAdapterFinetuneEngine(
                    train_dataset,
                    base_embed_model,
                    adapter_model=adapter_model(**adapter_config),  # I set the default parameters in my_adapter.py
                    model_output_path=model_output_path,
                    epochs=args.epochs,
                    verbose=True,
                    dim=1, # hack: dim is of no use if I pass the adapter_model, so just randomly set a number
                )
                
            # train
            wandb_callback = WandbCallback()
            finetune_engine.finetune(adapter_cls=adapter_model, callback=wandb_callback)
            embed_model = finetune_engine.get_finetuned_model(adapter_cls=adapter_model)
            end_time = time.time()
            training_time = end_time - start_time
            logger.info(f"Training finished in {training_time:.2f} seconds.")

            def get_dir_size_mb(dir_path):
                total_size = 0
                for dirpath, _, filenames in os.walk(dir_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.isfile(fp):
                            total_size += os.path.getsize(fp)
                return total_size / (1024 * 1024)

            model_size_mb = get_dir_size_mb(model_output_path)
            logger.info(f"Model saved to {model_output_path}. Model size ~ {model_size_mb:.2f} MB.")
            
        except RuntimeError as re:
            logger.error(f"Training failed due to runtime error (possibly GPU OOM): {re}")
            logger.debug(traceback.format_exc())
            return
        except Exception as e:
            logger.error(f"Training failed due to unexpected error: {e}")
            logger.debug(traceback.format_exc())
            return

    if "eval" in args.modes:
        if not os.path.exists(model_output_path):
            logger.warning(f"No finetuned model found at {model_output_path}.")
            return
        
        
        logger.info("Start evaluation...")
        try:
            local_model_id = model_output_path
            if not isinstance(local_model_id, str) or not local_model_id.startswith("local:"):
                local_model_id = f"local:{local_model_id}"
                
            # evaluate
            if args.adapter_model is None:
                eval_results_finetuned, hit_rate = evaluate(eval_dataset, local_model_id)
                eval_name = f"eval_other_{model_output_name}"
                st_evaluator_result = evaluate_st(eval_dataset, model=None, model_id=model_output_path, name=eval_name, output_path=args.result_output_path)
            else:
                if "train" not in args.modes:
                    base_embed_model = resolve_embed_model("local:" + base_model_id)
                    logger.info(base_embed_model)
                    adapter_model = adapter_model_dict[args.adapter_model]
                    adapter_config = adapter_config_dict[args.adapter_model][args.model_version]
                    finetune_engine = MyEmbeddingAdapterFinetuneEngine(
                        train_dataset,
                        base_embed_model,
                        adapter_model=adapter_model(**adapter_config),
                        model_output_path=model_output_path,
                        epochs=args.epochs,
                        verbose=True,
                        dim=1,
                    )
                    embed_model = finetune_engine.get_finetuned_model(adapter_cls=adapter_model)
                else:
                    embed_model = finetune_engine.get_finetuned_model(adapter_cls=adapter_model_dict[args.adapter_model])
                
                # eval by retriever
                eval_results_finetuned, hit_rate = evaluate(eval_dataset, embed_model)
                # eval by similarity scores
                eval_name = f"eval_other_{model_output_name}"
                wrapped_embed_model = MySentenceTransformerWrapper(embed_model)
                st_evaluator_result = evaluate_st(eval_dataset, model=wrapped_embed_model, model_id=base_model_id, name=eval_name, output_path=args.result_output_path)

            
            # write to file
            df_finetuned = pd.DataFrame(eval_results_finetuned)
            custom_eval_result_path = os.path.join(args.result_output_path, f"eval_details_{model_output_name}.csv")
            df_finetuned.to_csv(custom_eval_result_path, index=False)
            logger.info(f"Custom evaluation results saved to {custom_eval_result_path}")
            
            logger.info("InformationRetrievalEvaluator result\n", st_evaluator_result)
            METRIC_COLUMNS = [
                "cosine-Accuracy@1", "cosine-Accuracy@3", "cosine-Accuracy@5", "cosine-Accuracy@10",
                "cosine-Precision@1", "cosine-Precision@3", "cosine-Precision@5", "cosine-Precision@10",
                "cosine-Recall@1", "cosine-Recall@3", "cosine-Recall@5", "cosine-Recall@10",
                "cosine-MRR@10", "cosine-NDCG@10", "cosine-MAP@100",
            ]
            extracted_metrics = [[], []]
            for metric_suffix in METRIC_COLUMNS:
                matched_value = None
                for k, v in st_evaluator_result.items():
                    if metric_suffix in k:
                        matched_value = float(v)
                        break
                extracted_metrics[0].append(metric_suffix)
                extracted_metrics[1].append(matched_value if matched_value is not None else None)
            extracted_metrics[0].append("hit_rate")
            extracted_metrics[1].append(hit_rate)
            eval_result_path = os.path.join(args.result_output_path, f"eval_metrics_{model_output_name}.csv")
            write_tocsv(file_path=eval_result_path, row_data=extracted_metrics)
            logger.info(f"Evaluation finished. Results have been saved and logged in {eval_result_path}.")
            
        except RuntimeError as re:
            logger.error(f"Evaluation failed due to runtime error (possibly GPU OOM): {re}")
            logger.debug(traceback.format_exc())
            return
        except Exception as e:
            logger.error(f"Evaluation failed due to unexpected error: {e}")
            logger.debug(traceback.format_exc())
            return
    logger.info("All requested modes finished.")

if __name__ == "__main__":
    main()
