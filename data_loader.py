import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
import io
import base64
from typing import Optional, Dict, Any, List, Tuple
import random

try:
    from datasets import load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False
    print("[WARNING] 'datasets' library not available. Using synthetic data.")


class BaseDataset(Dataset):
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-0.5B", max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _mask_padding_in_labels(self, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Set labels to -100 where attention_mask is 0 (padding tokens)."""
        labels = labels.clone()
        labels[attention_mask == 0] = -100
        return labels


class CodeDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str = "bigcode/starcoderdata",
        language: str = "python",
        split: str = "train",
        max_length: int = 2048,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        num_samples: Optional[int] = None,
        streaming: bool = True
    ):
        super().__init__(tokenizer_name, max_length)
        
        self.language = language
        self.streaming = streaming
        self.num_samples = num_samples or 100
        self._cached_data = []
        
        raw_texts = []
        # Try to load real data first, fall back to synthetic
        if HAS_HF_DATASETS:
            try:
                raw_texts = self._load_real_data(dataset_name, language, split)
            except Exception as e:
                print(f"[WARNING] Failed to load {dataset_name}: {e}")
                print("[WARNING] Falling back to synthetic data.")
                raw_texts = self._generate_synthetic_code()
        else:
            raw_texts = self._generate_synthetic_code()
            
        self._process_chunks(raw_texts)
    
    def _load_real_data(self, dataset_name, language, split):
        """Load real code data from HuggingFace."""
        print(f"[INFO] Loading {dataset_name} (language={language})...")
        texts = []
        if self.streaming:
            ds = load_dataset(dataset_name, data_dir=language, split=split, streaming=True)
            for i, sample in enumerate(ds):
                if i >= self.num_samples:
                    break
                texts.append(sample.get('content', ''))
        else:
            ds = load_dataset(dataset_name, data_dir=language, split=split)
            for i in range(min(self.num_samples, len(ds))):
                texts.append(ds[i].get('content', ''))
        
        if not texts:
            raise ValueError("No data loaded from dataset")
        print(f"[INFO] Loaded {len(texts)} real samples from {dataset_name}")
        return texts
    
    def _generate_synthetic_code(self):
        """Generate synthetic code samples as fallback."""
        code_samples = [
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n    \n    def append(self, val):\n        if not self.head:\n            self.head = ListNode(val)\n        else:\n            curr = self.head\n            while curr.next:\n                curr = curr.next\n            curr.next = ListNode(val)",
            "import numpy as np\n\ndef matrix_multiply(A, B):\n    result = np.zeros((A.shape[0], B.shape[1]))\n    for i in range(A.shape[0]):\n        for j in range(B.shape[1]):\n            for k in range(A.shape[1]):\n                result[i, j] += A[i, k] * B[k, j]\n    return result",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
            "class Stack:\n    def __init__(self):\n        self.items = []\n    \n    def push(self, item):\n        self.items.append(item)\n    \n    def pop(self):\n        if not self.is_empty():\n            return self.items.pop()\n    \n    def is_empty(self):\n        return len(self.items) == 0",
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        ]
        
        texts = []
        for i in range(self.num_samples):
            texts.append(random.choice(code_samples))
        print(f"[INFO] Generated {self.num_samples} synthetic code samples")
        return texts

    def _process_chunks(self, texts):
        self._cached_data = []
        for text in texts:
            if not text.endswith(self.tokenizer.eos_token):
                text = text + self.tokenizer.eos_token
                
            encoding = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
            input_ids_full = encoding['input_ids'].squeeze(0)
            
            seq_len = input_ids_full.size(0)
            for i in range(0, seq_len, self.max_length):
                chunk = input_ids_full[i : i + self.max_length]
                pad_len = self.max_length - chunk.size(0)
                
                if pad_len > 0:
                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                    padded_chunk = torch.cat([chunk, torch.full((pad_len,), pad_id, dtype=torch.long)])
                    attention_mask = torch.cat([torch.ones_like(chunk), torch.zeros(pad_len, dtype=torch.long)])
                else:
                    padded_chunk = chunk
                    attention_mask = torch.ones_like(chunk)
                
                labels = self._mask_padding_in_labels(padded_chunk, attention_mask)
                self._cached_data.append({
                    'input_ids': padded_chunk,
                    'attention_mask': attention_mask,
                    'labels': labels
                })
        print(f"[INFO] Processed texts into {len(self._cached_data)} chunks of length {self.max_length}")
    
    def __len__(self):
        return len(self._cached_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._cached_data[idx % len(self._cached_data)]


class CodeInstructionDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str = "sahil2801/code_alpaca",
        split: str = "train",
        max_length: int = 2048,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        num_samples: Optional[int] = None
    ):
        super().__init__(tokenizer_name, max_length)
        
        self._cached_data = []
        
        if HAS_HF_DATASETS:
            try:
                data = load_dataset(dataset_name, split=split)
                if num_samples:
                    data = data.select(range(min(num_samples, len(data))))
                self._cached_data = list(data)
                print(f"[INFO] Loaded {len(self._cached_data)} instruction samples from {dataset_name}")
            except Exception as e:
                print(f"[WARNING] Failed to load {dataset_name}: {e}")
                print("[WARNING] Falling back to synthetic instruction data.")
                self._generate_synthetic_instructions()
        else:
            self._generate_synthetic_instructions()
    
    def _generate_synthetic_instructions(self):
        """Generate synthetic instruction-response pairs as fallback."""
        instruction_pairs = [
            {
                'instruction': 'Write a function that reverses a string.',
                'input': '',
                'output': 'def reverse_string(s):\n    return s[::-1]'
            },
            {
                'instruction': 'Write a function to check if a number is even.',
                'input': '',
                'output': 'def is_even(n):\n    return n % 2 == 0'
            },
            {
                'instruction': 'Write a function to find the maximum element in a list.',
                'input': '',
                'output': 'def find_max(lst):\n    if not lst:\n        return None\n    max_val = lst[0]\n    for item in lst[1:]:\n        if item > max_val:\n            max_val = item\n    return max_val'
            },
            {
                'instruction': 'Write a function that counts the number of vowels in a string.',
                'input': '',
                'output': 'def count_vowels(s):\n    return sum(1 for c in s.lower() if c in "aeiou")'
            },
            {
                'instruction': 'Write a function to merge two sorted lists.',
                'input': '',
                'output': 'def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result'
            },
            {
                'instruction': 'Write a Python class for a simple calculator.',
                'input': '',
                'output': 'class Calculator:\n    def add(self, a, b):\n        return a + b\n    def subtract(self, a, b):\n        return a - b\n    def multiply(self, a, b):\n        return a * b\n    def divide(self, a, b):\n        if b == 0:\n            raise ValueError("Cannot divide by zero")\n        return a / b'
            },
            {
                'instruction': 'Implement a function to flatten a nested list.',
                'input': '',
                'output': 'def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result'
            },
            {
                'instruction': 'Write a function to compute the GCD of two numbers.',
                'input': '',
                'output': 'def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a'
            },
        ]
        
        num_samples = getattr(self, 'num_samples', None) or 100
        self._cached_data = []
        for i in range(num_samples):
            self._cached_data.append(random.choice(instruction_pairs))
        print(f"[INFO] Generated {len(self._cached_data)} synthetic instruction samples")
    
    def __len__(self):
        return len(self._cached_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._cached_data[idx % len(self._cached_data)]
        
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        # Build prompt — loss should only be on the response
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        full_text = prompt + output + self.tokenizer.eos_token
        
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Mask prompt tokens from the loss
        prompt_encoding = self.tokenizer(prompt, return_tensors='pt')
        prompt_len = prompt_encoding['input_ids'].shape[1]
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        mask_len = min(prompt_len, self.max_length)
        labels[:mask_len] = -100  # Don't compute loss on prompt
        labels = self._mask_padding_in_labels(labels, attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def create_ocr_dataset(
    max_length: int = 2048,
    image_size: int = 336,
    num_samples: int = 5000,
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
    streaming: bool = True
) -> Dataset:
    return OCRDataset(
        max_length=max_length,
        image_size=image_size,
        num_samples=num_samples,
        tokenizer_name=tokenizer_name,
        use_synthetic=True
    )


