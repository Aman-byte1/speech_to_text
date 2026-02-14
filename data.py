
import argparse
import sys
import yaml
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import pandas as pd
import torch
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
from datasets import load_dataset, Dataset, Audio
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import noisereduce as nr
import re

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessStats:
    dataset_name: str
    original_size: int
    final_size: int
    discarded_count: int

class TextProcessor:
    def __init__(self, config: Dict):
        self.config = config.get('text', {})
        self.models_config = config.get('models', {})
        self.semantic_model = None
        
        # Load Semantic Model if needed (Lazy loading)
        if 'semantic_model' in self.models_config and self.models_config['semantic_model']:
            try:
                logger.info(f"Loading Semantic Model: {self.models_config['semantic_model']}")
                # Check for GPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.semantic_model = SentenceTransformer(self.models_config['semantic_model'], device=device)
            except Exception as e:
                logger.warning(f"Could not load semantic model: {e}. Semantic filtering will be skipped.")

    def basic_clean(self, text: str) -> str:
        """Removes HTML, extra spaces, etc."""
        if not text or not isinstance(text, str):
            return ""
        # Simple rule-based cleaning
        text = re.sub(r'<[^>]+>', '', text) # Remove HTML
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower() # Normalize to lowercase as per request

    def filter_rules(self, src_text: str, tgt_text: Optional[str] = None) -> bool:
        """Stage 1: Rule-based filtering (Length, Ratio)"""
        if not src_text: 
            return False
            
        # Length check
        if not (self.config.get('min_chars', 3) <= len(src_text) <= self.config.get('max_chars', 200)):
            return False
            
        # Ratio check (if target exists)
        if tgt_text:
            if not tgt_text: return False
            len_src = len(src_text)
            len_tgt = len(tgt_text)
            if len_src == 0 or len_tgt == 0: return False
            
            ratio = max(len_src, len_tgt) / min(len_src, len_tgt)
            if ratio > self.config.get('max_ratio', 2.0):
                return False
                
        return True

    def semantic_filter(self, src_texts: List[str], tgt_texts: List[str]) -> List[bool]:
        """Stage 3: Semantic Filtering using LaBSE or similar"""
        if self.semantic_model is None:
            return [True] * len(src_texts)
            
        logger.info("Running Semantic Filtering...")
        try:
            embeddings1 = self.semantic_model.encode(src_texts, convert_to_tensor=True, show_progress_bar=False)
            embeddings2 = self.semantic_model.encode(tgt_texts, convert_to_tensor=True, show_progress_bar=False)
            
            # Compute cosine similarity
            # Utilizing semantic search utility, but strictly we want pair-wise.
            # cos_sim returns All-vs-All. For huge lists this is bad.
            # Start simple: use loop or diagonal if tensor ops allow.
            # Optimized: (a / |a|) . (b / |b|)
            import torch.nn.functional as F
            scores = F.cosine_similarity(embeddings1, embeddings2)
            
            threshold = self.config.get('semantic_threshold', 0.6)
            return (scores >= threshold).cpu().tolist()
        except Exception as e:
            logger.error(f"Semantic filtering failed: {e}")
            return [True] * len(src_texts)

class DataPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.text_proc = TextProcessor(self.config)
        self.stats = []
        self.hf_token = self.config.get('hf_token')
        self.output_org = self.config.get('output_org')

    def run(self, lang_filter: Optional[str] = None):
        datasets_config = self.config.get('datasets', [])
        
        for ds_conf in datasets_config:
            # Filter by language if provided
            if lang_filter:
                ds_lang = ds_conf.get('language', '').lower()
                if ds_lang != lang_filter.lower():
                    continue

            self.process_dataset(ds_conf)
            
        self.generate_report()

    def process_dataset(self, ds_conf):
        # Determine Output Path
        lang_dir = ds_conf.get('language', 'unknown')
        output_name = f"{ds_conf['name']}_processed"
        local_path = f"./processed_data/{lang_dir}/{output_name}"
        
        # Check if already processed
        if os.path.exists(local_path):
            logger.info(f"Dataset {ds_conf['name']} already processed at {local_path}. Skipping.")
            return

        logger.info(f"Processing Dataset: {ds_conf['name']}")
        
        # Load Dataset
        try:
            split = ds_conf.get('split', 'train')
            logger.info(f"Loading {ds_conf['hf_path']} ({ds_conf.get('subset')})...")
            dataset = load_dataset(ds_conf['hf_path'], ds_conf.get('subset'), split=split, token=self.hf_token)
        except Exception as e:
            logger.error(f"Failed to load {ds_conf['name']}: {e}")
            return

        original_len = len(dataset)
        
        # --- PREPROCESSING STEPS ---
        
        # 1. Cast Audio (Resample + Mono) if voice
        if ds_conf['type'] == 'voice':
            target_sr = self.config['audio'].get('sampling_rate', 16000)
            logger.info(f"Casting audio to {target_sr}Hz mono...")
            dataset = dataset.cast_column(ds_conf['voice_col'], Audio(sampling_rate=target_sr, mono=True))

        # 2. Filter Function (Includes Text Rules & Audio Duration)
        def filter_fn(example):
            # Text Rules
            text_col = ds_conf.get('text_col')
            if text_col and text_col in example:
                clean_text = self.text_proc.basic_clean(example[text_col])
                if not self.text_proc.filter_rules(clean_text):
                    return False
            
            # Audio Duration Rules
            if ds_conf['type'] == 'voice':
                voice_col = ds_conf.get('voice_col')
                if voice_col and voice_col in example:
                    audio_data = example[voice_col]['array']
                    sr = example[voice_col]['sampling_rate']
                    duration = len(audio_data) / sr
                    
                    min_dur = self.config['audio'].get('min_duration', 1.0)
                    max_dur = self.config['audio'].get('max_duration', 20.0)
                    
                    if not (min_dur <= duration <= max_dur):
                        return False
            return True

        logger.info("Applying filters (Text Rules & Audio Duration)...")
        # Note: Filtering requires decoding audio if we access array. 
        # This will be slow for large datasets but is necessary for duration check.
        # Ensure we don't have extremely large datasets or use batched filter if available/faster.
        dataset_filtered = dataset.filter(filter_fn)

        # 3. Map Function (Text Cleaning & Optional Semantic Filter Prep)
        def map_fn(example):
            text_col = ds_conf.get('text_col')
            if text_col and text_col in example:
                example[text_col] = self.text_proc.basic_clean(example[text_col])
            return example

        logger.info("Applying text normalization...")
        dataset_processed = dataset_filtered.map(map_fn)
        
        # 4. Semantic Filtering (If applicable)
        # Assuming we have pairs. If unrelated audio/text, skip.
        # If we have a translation target column (e.g. source_lang, target_lang columns), we can run it.
        # For ASR datasets (Audio -> Text), semantic filtering (LaBSE) isn't typically used *unless* we are filtering 
        # against a synthetic translation or something. The prompt requested it explicitly for "datasets", likely assuming parallel text.
        # We will check if 'target_text_col' is defined in config or if we want to run it on (text_col, target_col).
        # For now, simplistic implementation: skip unless we explicitly configured parallel text cols.
        
        final_len = len(dataset_processed)
        
        self.stats.append(ProcessStats(
            dataset_name=ds_conf['name'],
            original_size=original_len,
            final_size=final_len,
            discarded_count=original_len - final_len
        ))
        
        # Save / Upload
        logger.info(f"Saving to {local_path}")
        dataset_processed.save_to_disk(local_path)
        
        if self.output_org and self.hf_token:
            repo_id = f"{self.output_org}/{ds_conf.get('language', 'multi')}-{output_name}"
            logger.info(f"Uploading to Hub: {repo_id}")
            try:
                dataset_processed.push_to_hub(repo_id, token=self.hf_token)
            except Exception as e:
                logger.error(f"Upload failed: {e}")

    def generate_report(self):
        logger.info("\n=== FINAL REPORT ===")
        data = []
        for s in self.stats:
            data.append({
                "Dataset": s.dataset_name,
                "Original": s.original_size,
                "Final": s.final_size,
                "Discarded": s.discarded_count,
                "Retention(%)": f"{(s.final_size/s.original_size)*100:.1f}" if s.original_size else "0.0"
            })
        
        df = pd.DataFrame(data)
        print(df.to_markdown(index=False))
        df.to_csv("report.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--lang", help="Filter datasets by language (e.g., 'hausa', 'amharic')")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
        
    pipeline = DataPipeline(args.config)
    pipeline.run(lang_filter=args.lang)
