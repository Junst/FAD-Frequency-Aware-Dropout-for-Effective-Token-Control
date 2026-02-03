# Tag Frequency-based Adaptive Dropout for DiffSynth-Studio
# Based on Frequency-Aware Dropout (kohya-ss/sd-scripts adaptation)
# 
# Key idea: Tags that appear frequently (e.g., "1girl", "smile") are more 
# likely to be dropped, while rare tags (e.g., specific character names) 
# are preserved. This helps the model learn rare concepts better.
#
# sFAD (Step-based FAD): FAD gradually increases over training steps.
# pstep(i) starts from near zero and exponentially increases to 1,
# thereby applying almost no dropout in early iterations and complete 
# dropout toward the end of training.

import math
import random
from typing import Dict, List, Optional, Set
import json


class TagFrequencyDropout:
    """
    Tag Frequency-based Adaptive Dropout.
    
    Computes dropout probability for each tag based on its frequency in the dataset.
    Frequent tags get higher dropout probability, rare tags get lower probability.
    
    Args:
        min_dropout_rate: Minimum dropout probability for rare tags (default: 0.0)
        max_dropout_rate: Maximum dropout probability for frequent tags (default: 0.5)
        sigmoid_alpha: Controls how steep the sigmoid curve is (default: 10.0)
        sigmoid_center: The frequency ratio around which sigmoid transitions (default: 0.5)
        trigger_tokens: Set of tokens that should never be dropped (e.g., character name)
        no_dropout_tokens: Additional tokens that should never be dropped
        shuffle: Whether to shuffle tags after dropout (default: True)
    """
    
    def __init__(
        self,
        min_dropout_rate: float = 0.0,
        max_dropout_rate: float = 0.5,
        sigmoid_alpha: float = 10.0,
        sigmoid_center: float = 0.5,
        trigger_tokens: Optional[Set[str]] = None,
        no_dropout_tokens: Optional[Set[str]] = None,
        shuffle: bool = True,
        # sFAD (Step-based FAD) parameters
        step_based_dropout: bool = False,
        step_dropout_schedule: str = "exponential",  # linear, cosine, exponential, exp_up, jump_ramp
        step_dropout_start: float = 0.0,  # pstep starts near 0
        step_dropout_end: float = 1.0,    # pstep ends at 1 (full FAD)
        step_dropout_warmup_ratio: float = 0.1,  # warmup period
        use_warmup: bool = True,
    ):
        self.min_dropout_rate = min_dropout_rate
        self.max_dropout_rate = max_dropout_rate
        self.sigmoid_alpha = sigmoid_alpha
        self.sigmoid_center = sigmoid_center
        self.trigger_tokens = trigger_tokens or set()
        self.no_dropout_tokens = no_dropout_tokens or set()
        self.shuffle = shuffle
        
        # sFAD (Step-based FAD) parameters
        self.step_based_dropout = step_based_dropout
        self.step_dropout_schedule = step_dropout_schedule
        self.step_dropout_start = step_dropout_start
        self.step_dropout_end = step_dropout_end
        self.step_dropout_warmup_ratio = step_dropout_warmup_ratio
        self.use_warmup = use_warmup
        
        # Tag statistics
        self.tag_frequency: Dict[str, int] = {}
        self.total_captions: int = 0
        self.dropout_prob: Dict[str, float] = {}
        
        # Statistics for logging
        self.dropped_count: Dict[str, int] = {}
        self.kept_count: Dict[str, int] = {}
    
    def add_captions(self, captions: List[str], delimiter: str = ","):
        """
        Add captions to compute tag frequencies.
        
        Args:
            captions: List of caption strings
            delimiter: Character used to separate tags (default: ",")
        """
        self.total_captions += len(captions)
        
        for caption in captions:
            tags = caption.split(delimiter)
            for tag in tags:
                tag = tag.strip().lower()
                if tag:
                    self.tag_frequency[tag] = self.tag_frequency.get(tag, 0) + 1
        
        # Recalculate dropout probabilities
        self._calculate_dropout_probs()
    
    def add_captions_from_metadata(self, metadata: List[dict], caption_key: str = "prompt", delimiter: str = ","):
        """
        Add captions from metadata list (common format in DiffSynth-Studio).
        
        Args:
            metadata: List of dicts containing captions
            caption_key: Key to access caption in each dict (default: "prompt")
            delimiter: Character used to separate tags
        """
        captions = []
        for item in metadata:
            if caption_key in item:
                captions.append(str(item[caption_key]))
        self.add_captions(captions, delimiter)
    
    def _calculate_dropout_probs(self):
        """
        Compute dropout probability for each tag based on frequency.
        Uses sigmoid function for smooth transition.
        
        - ratio = tag_count / total_captions
        - logistic = 1 / (1 + exp(-alpha * (ratio - center)))
        - dropout = min_rate + (max_rate - min_rate) * logistic
        """
        if self.total_captions == 0:
            return
        
        for tag, count in self.tag_frequency.items():
            ratio = count / self.total_captions
            
            # Sigmoid function for smooth transition
            logistic = 1.0 / (1.0 + math.exp(-self.sigmoid_alpha * (ratio - self.sigmoid_center)))
            
            # Linear interpolation between min and max dropout rates
            dropout = self.min_dropout_rate + (self.max_dropout_rate - self.min_dropout_rate) * logistic
            
            # Clamp to [0, 1]
            self.dropout_prob[tag] = max(0.0, min(1.0, dropout))
    
    def _should_keep_tag(self, tag: str) -> bool:
        """Check if tag should be kept (not dropped)."""
        tag_lower = tag.lower()
        
        # Check trigger tokens (never drop)
        for trigger in self.trigger_tokens:
            if trigger.lower() in tag_lower:
                return True
        
        # Check no_dropout_tokens
        if tag_lower in self.no_dropout_tokens:
            return True
        
        return False
    
    def get_step_based_dropout_prob(self, current_step: int, max_train_steps: int) -> float:
        """
        Calculate step-based dropout multiplier for sFAD.
        
        In sFAD, the dropout probability gradually increases over training:
        - Early iterations: pstep ≈ 0 (almost no dropout)
        - End of training: pstep ≈ 1 (full FAD dropout)
        
        The final dropout for a tag = adapt_prob * pstep
        
        Args:
            current_step: Current training step
            max_train_steps: Total training steps
        
        Returns:
            Step-based dropout probability multiplier [0, 1]
        """
        if not self.step_based_dropout or max_train_steps <= 0:
            return 1.0  # No step modulation, use full FAD
        
        p = current_step / max_train_steps
        start = self.step_dropout_start
        end = self.step_dropout_end
        k = 5.0  # Exponential decay/growth rate
        sched = self.step_dropout_schedule
        use_wu = self.use_warmup
        wu_st = int(max_train_steps * self.step_dropout_warmup_ratio)
        
        if sched == "jump_ramp":
            # Jump to end value immediately after warmup
            if use_wu:
                if current_step <= wu_st:
                    ratio = current_step / wu_st
                    dropout_rate = start + (end - start) * ratio
                else:
                    dropout_rate = end
            else:
                dropout_rate = end
        
        elif sched == "exp_up":
            # Exponential increase (for sFAD: starts low, ends high)
            if use_wu and current_step < wu_st:
                dropout_rate = start
            else:
                progress = p if not use_wu else (current_step - wu_st) / max(max_train_steps - wu_st, 1)
                inc = 1 - math.exp(-k * progress)
                dropout_rate = start + (end - start) * inc
        
        else:
            # linear, cosine, exponential schedules
            if use_wu and current_step < wu_st:
                dropout_rate = start
            else:
                progress = p if not use_wu else (current_step - wu_st) / max(max_train_steps - wu_st, 1)
                if sched == "linear":
                    dropout_rate = start + (end - start) * progress
                elif sched == "cosine":
                    cos_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
                    dropout_rate = start + (end - start) * cos_factor
                elif sched == "exponential":
                    # Exponential increase from start to end
                    decay_factor = 1 - math.exp(-5 * progress)
                    dropout_rate = start + (end - start) * decay_factor
                else:
                    dropout_rate = start + (end - start) * progress
        
        return max(0.0, min(1.0, dropout_rate))
    
    def process_caption(
        self, 
        caption: str, 
        delimiter: str = ",",
        current_step: Optional[int] = None,
        max_train_steps: Optional[int] = None,
    ) -> str:
        """
        Apply frequency-based dropout to a caption.
        
        For sFAD (step-based FAD), the final dropout probability is:
            final_prob = adapt_prob * step_prob
        where step_prob gradually increases from 0 to 1 over training.
        
        Args:
            caption: Input caption string
            delimiter: Character used to separate tags
            current_step: Current training step (for sFAD)
            max_train_steps: Total training steps (for sFAD)
        
        Returns:
            Processed caption with some tags dropped
        """
        tags = caption.split(delimiter)
        kept_tags = []
        
        # Get step-based multiplier for sFAD
        if self.step_based_dropout and current_step is not None and max_train_steps is not None:
            step_prob = self.get_step_based_dropout_prob(current_step, max_train_steps)
        else:
            step_prob = 1.0  # No step modulation
        
        for tag in tags:
            original_tag = tag.strip()
            tag_lower = original_tag.lower()
            
            if not tag_lower:
                continue
            
            # Check if tag should never be dropped
            if self._should_keep_tag(tag_lower):
                kept_tags.append(original_tag)
                self.kept_count[tag_lower] = self.kept_count.get(tag_lower, 0) + 1
                continue
            
            # Get frequency-based dropout probability
            adapt_prob = self.dropout_prob.get(tag_lower, 0.0)
            
            # sFAD: final_prob = adapt_prob * step_prob
            # At early steps, step_prob ≈ 0, so almost no dropout
            # At late steps, step_prob ≈ 1, so full FAD dropout
            final_prob = adapt_prob * step_prob
            
            # Decide whether to keep
            if random.random() > final_prob:
                kept_tags.append(original_tag)
                self.kept_count[tag_lower] = self.kept_count.get(tag_lower, 0) + 1
            else:
                self.dropped_count[tag_lower] = self.dropped_count.get(tag_lower, 0) + 1
        
        # Shuffle if enabled
        if self.shuffle:
            random.shuffle(kept_tags)
        
        return f"{delimiter} ".join(kept_tags)
    
    def get_stats(self) -> Dict:
        """Get statistics about tag frequencies and dropout rates."""
        stats = {
            "total_captions": self.total_captions,
            "unique_tags": len(self.tag_frequency),
            "config": {
                "min_dropout_rate": self.min_dropout_rate,
                "max_dropout_rate": self.max_dropout_rate,
                "sigmoid_alpha": self.sigmoid_alpha,
                "sigmoid_center": self.sigmoid_center,
            }
        }
        
        # Top 10 most frequent tags with their dropout rates
        sorted_tags = sorted(self.tag_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        stats["top_frequent_tags"] = [
            {"tag": tag, "count": count, "ratio": count/self.total_captions if self.total_captions > 0 else 0, 
             "dropout_prob": self.dropout_prob.get(tag, 0)}
            for tag, count in sorted_tags
        ]
        
        # Top 10 rarest tags
        sorted_rare = sorted(self.tag_frequency.items(), key=lambda x: x[1])[:10]
        stats["top_rare_tags"] = [
            {"tag": tag, "count": count, "ratio": count/self.total_captions if self.total_captions > 0 else 0,
             "dropout_prob": self.dropout_prob.get(tag, 0)}
            for tag, count in sorted_rare
        ]
        
        return stats
    
    def save(self, path: str):
        """Save frequency data to file."""
        data = {
            "tag_frequency": self.tag_frequency,
            "total_captions": self.total_captions,
            "dropout_prob": self.dropout_prob,
            "config": {
                "min_dropout_rate": self.min_dropout_rate,
                "max_dropout_rate": self.max_dropout_rate,
                "sigmoid_alpha": self.sigmoid_alpha,
                "sigmoid_center": self.sigmoid_center,
                # sFAD config
                "step_based_dropout": self.step_based_dropout,
                "step_dropout_schedule": self.step_dropout_schedule,
                "step_dropout_start": self.step_dropout_start,
                "step_dropout_end": self.step_dropout_end,
                "step_dropout_warmup_ratio": self.step_dropout_warmup_ratio,
                "use_warmup": self.use_warmup,
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load frequency data from file."""
        with open(path, "r") as f:
            data = json.load(f)
        self.tag_frequency = data["tag_frequency"]
        self.total_captions = data["total_captions"]
        self.dropout_prob = data["dropout_prob"]
        if "config" in data:
            self.min_dropout_rate = data["config"].get("min_dropout_rate", self.min_dropout_rate)
            self.max_dropout_rate = data["config"].get("max_dropout_rate", self.max_dropout_rate)
            self.sigmoid_alpha = data["config"].get("sigmoid_alpha", self.sigmoid_alpha)
            self.sigmoid_center = data["config"].get("sigmoid_center", self.sigmoid_center)
            # sFAD config
            self.step_based_dropout = data["config"].get("step_based_dropout", self.step_based_dropout)
            self.step_dropout_schedule = data["config"].get("step_dropout_schedule", self.step_dropout_schedule)
            self.step_dropout_start = data["config"].get("step_dropout_start", self.step_dropout_start)
            self.step_dropout_end = data["config"].get("step_dropout_end", self.step_dropout_end)
            self.step_dropout_warmup_ratio = data["config"].get("step_dropout_warmup_ratio", self.step_dropout_warmup_ratio)
            self.use_warmup = data["config"].get("use_warmup", self.use_warmup)


class TagFrequencyDropoutDataset:
    """
    Wrapper that applies Tag Frequency Dropout to a dataset.
    
    For sFAD (step-based FAD), call set_training_progress() each step to update
    the step-based dropout multiplier.
    
    Usage:
        dataset = UnifiedDataset(...)
        dropout = TagFrequencyDropout(step_based_dropout=True, ...)  # for sFAD
        wrapped = TagFrequencyDropoutDataset(dataset, dropout, caption_key="prompt")
        
        # In training loop:
        wrapped.set_training_progress(current_step, max_train_steps)
    """
    
    def __init__(
        self,
        dataset,
        tag_dropout: TagFrequencyDropout,
        caption_key: str = "prompt",
        delimiter: str = ",",
        max_train_steps: Optional[int] = None,
    ):
        self.dataset = dataset
        self.tag_dropout = tag_dropout
        self.caption_key = caption_key
        self.delimiter = delimiter
        
        # sFAD step tracking
        self.current_step = 0
        self.max_train_steps = max_train_steps
        
        # Pre-compute tag frequencies from dataset metadata
        if hasattr(dataset, 'data') and dataset.data:
            self.tag_dropout.add_captions_from_metadata(
                dataset.data, 
                caption_key=caption_key,
                delimiter=delimiter
            )
            print(f"Tag Frequency Dropout initialized with {len(dataset.data)} captions")
            stats = self.tag_dropout.get_stats()
            print(f"  - Unique tags: {stats['unique_tags']}")
            if stats['top_frequent_tags']:
                top_tag = stats['top_frequent_tags'][0]
                print(f"  - Most frequent: '{top_tag['tag']}' ({top_tag['count']} times, dropout: {top_tag['dropout_prob']:.2%})")
            if self.tag_dropout.step_based_dropout:
                print(f"  - sFAD enabled: schedule={self.tag_dropout.step_dropout_schedule}, "
                      f"start={self.tag_dropout.step_dropout_start}, end={self.tag_dropout.step_dropout_end}")
    
    def set_training_progress(self, current_step: int, max_train_steps: Optional[int] = None):
        """
        Update training progress for sFAD.
        
        Call this each training step before fetching data.
        
        Args:
            current_step: Current training step
            max_train_steps: Total training steps (optional, uses init value if not provided)
        """
        self.current_step = current_step
        if max_train_steps is not None:
            self.max_train_steps = max_train_steps
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # Apply dropout to caption
        if self.caption_key in data and isinstance(data[self.caption_key], str):
            original_caption = data[self.caption_key]
            data[self.caption_key] = self.tag_dropout.process_caption(
                original_caption, 
                delimiter=self.delimiter,
                current_step=self.current_step,
                max_train_steps=self.max_train_steps,
            )
        
        return data
    
    def __len__(self):
        return len(self.dataset)
    
    def __getattr__(self, name):
        """Proxy all other attributes to the underlying dataset."""
        # Avoid infinite recursion for our own attributes
        if name in ('dataset', 'tag_dropout', 'caption_key', 'delimiter', 
                    'current_step', 'max_train_steps'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.dataset, name)


