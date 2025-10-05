import os
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import matplotlib.pyplot as plt

def convert_sharegpt_to_standard(example):
    """Convert ShareGPT format to standard format"""
    conversations = example["conversations"]
    messages = []
    
    for conv in conversations:
        if conv["from"] == "human":
            messages.append({"role": "user", "content": conv["value"]})
        elif conv["from"] == "gpt":
            messages.append({"role": "assistant", "content": conv["value"]})
    
    example["messages"] = messages
    return example

def apply_chat_template(example, tokenizer):
    """Apply chat template to convert messages to text format"""
    example["text"] = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)
    return example

def tokenize_function(examples, tokenizer, max_seq_length):
    """Tokenize the text data"""
    tokenized = tokenizer(
        examples["text"],
        truncation=False,
        padding=False,
        return_overflowing_tokens=False,
        return_attention_mask=False,  # 不返回 attention_mask
    )
    # 临时添加 num_tokens 用于过滤，但不保存到最终数据集
    tokenized["num_tokens"] = [len(input_ids) for input_ids in tokenized["input_ids"]]
    
    return tokenized

def filter_by_length(example, max_seq_length):
    """Filter out examples that exceed maximum sequence length"""
    return example["num_tokens"] <= max_seq_length

def plot_token_distribution(all_token_counts, max_tokens, max_seq_length, output_dir):
    """Create and save token distribution plot"""
    print("Creating token distribution plot...")
    
    # Define bins with 256 intervals
    max_length = max(max_tokens, max_seq_length)
    bins = list(range(0, max_length + 256, 256))
    
    # Calculate interval percentages
    interval_percentages = []
    interval_labels = []
    
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        
        # Count tokens in this interval
        count = sum(1 for token_count in all_token_counts if bin_start < token_count <= bin_end)
        percentage = (count / len(all_token_counts)) * 100
        interval_percentages.append(percentage)
        interval_labels.append(f"{bin_start}-{bin_end}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    plt.bar(bin_centers, interval_percentages, width=200, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.xlabel('Token Count', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Token Length Distribution (Interval)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xlim(0, max_length)
    
    # Set x-axis ticks to show every 256
    plt.xticks(bins[::2])  # Show every other bin to avoid crowding
    
    # Save the plot
    plot_path = os.path.join(output_dir, "token_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Token distribution plot saved to: {plot_path}")
    
    return plot_path

def main():
    parser = argparse.ArgumentParser(description="Tokenize and save dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", 
                       help="Path to the model/tokenizer")
    parser.add_argument("--dataset_name", type=str, default="mlabonne/open-perfectblend",
                       help="Name of the dataset to load")
    parser.add_argument("--output_dir", type=str, default="./perfectblend",
                       help="Directory to save the processed dataset")
    parser.add_argument("--test_size", type=float, default=0.01,
                       help="Fraction of data to use for evaluation")
    parser.add_argument("--num_proc", type=int, default=32,
                       help="Number of processes for dataset processing")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length for tokenization")
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print(f"Loading dataset: {args.dataset_name}...")
    raw_dataset = load_dataset(args.dataset_name)
    train_dataset = raw_dataset["train"]
    column_names = list(train_dataset.features)
    
    print("Converting ShareGPT format to standard format...")
    converted_dataset = train_dataset.map(
        convert_sharegpt_to_standard,
        num_proc=args.num_proc,
        desc="Converting ShareGPT to standard format",
    )
    
    print("Applying chat template...")
    processed_dataset = converted_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Applying chat template",
    )
    
    print("Tokenizing dataset...")
    tokenized_dataset = processed_dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": args.max_seq_length},
        batched=True,
        num_proc=args.num_proc,
        remove_columns=["text"],  # Remove text column after tokenization
        desc="Tokenizing dataset",
    )
    
    print(f"Filtering out sequences longer than {args.max_seq_length} tokens...")
    original_size = len(tokenized_dataset)
    filtered_dataset = tokenized_dataset.filter(
        filter_by_length,
        fn_kwargs={"max_seq_length": args.max_seq_length},
        num_proc=args.num_proc,
        desc="Filtering by sequence length",
    )
    filtered_size = len(filtered_dataset)
    discarded_count = original_size - filtered_size
    print(f"Original dataset size: {original_size}")
    print(f"Filtered dataset size: {filtered_size}")
    print(f"Discarded {discarded_count} samples ({discarded_count/original_size*100:.2f}%)")
    
    print(f"Splitting dataset (test_size={args.test_size})...")
    train_test_split = filtered_dataset.train_test_split(test_size=args.test_size, seed=42)
    train_dataset_with_tokens = train_test_split['train']
    eval_dataset_with_tokens = train_test_split['test']
    
    print(f"Training samples: {len(train_dataset_with_tokens)}")
    print(f"Evaluation samples: {len(eval_dataset_with_tokens)}")
    
    # Calculate token statistics before removing num_tokens
    train_token_counts = list(train_dataset_with_tokens["num_tokens"])
    eval_token_counts = list(eval_dataset_with_tokens["num_tokens"])
    all_token_counts = train_token_counts + eval_token_counts
    
    # Remove num_tokens field from final datasets
    print("Removing num_tokens field from final dataset...")
    train_dataset = train_dataset_with_tokens.remove_columns(["num_tokens"])
    eval_dataset = eval_dataset_with_tokens.remove_columns(["num_tokens"])
    
    avg_tokens = sum(all_token_counts) / len(all_token_counts)
    min_tokens = min(all_token_counts)
    max_tokens = max(all_token_counts)
    
    print(f"Token statistics:")
    print(f"  Total tokens: {sum(all_token_counts)}")
    print(f"  Average tokens: {avg_tokens:.2f}")
    print(f"  Minimum tokens: {min_tokens}")
    print(f"  Maximum tokens: {max_tokens}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create token distribution plot
    plot_path = plot_token_distribution(all_token_counts, max_tokens, args.max_seq_length, args.output_dir)
    
    print(f"Saving processed dataset to {args.output_dir}...")
    # Save as a DatasetDict for easy loading
    dataset_dict = datasets.DatasetDict({
        'train': train_dataset,
        'eval': eval_dataset
    })
    dataset_dict.save_to_disk(args.output_dir)
    
    print("Dataset processing and saving completed!")
    print(f"You can now load the dataset using: datasets.load_from_disk('{args.output_dir}')")
    print("The dataset is now fully tokenized and ready for training!")

if __name__ == "__main__":
    main() 