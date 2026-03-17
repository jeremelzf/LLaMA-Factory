"""
Convert GraSP dataset from conversations/response format to ShareGPT format
for use with LLaMA-Factory and Qwen3-VL.
Splits dataset into training (80%) and evaluation (20%) sets, ensuring
samples from the same frame sequence stay together.
"""
import json
import os
import random
from collections import defaultdict

def convert_grasp_to_llamafactory(input_file, train_output_file, eval_output_file, include_general_response=True, train_ratio=0.8, random_seed=42):
    """
    Convert GraSP dataset to LLaMA-Factory format with question variations.
    Splits data into train/eval sets ensuring samples from same frame sequence stay together.
    
    Input format:
    {
        "id": 0,
        "images": ["path1.jpg", ...],
        "conversations": ["prompt1", "prompt2", ...],
        "response": "response text"
    }
    
    Output format:
    {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"}
        ],
        "images": ["path1.jpg", ...]
    }
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        grasp_data = json.load(f)
    
    # Group samples by frame sequence (same image paths)
    # Key: tuple of sorted image paths (normalized), Value: list of samples
    frame_sequence_groups = defaultdict(list)
    
    # Define question variations for diversity
    phase_step_questions = [
        "What surgical phase and step are shown in these images?",
        "Identify the surgical phase and step being performed.",
        "What surgical phase and step can you identify in these images?",
        "Describe the surgical phase and step visible in this sequence.",
        "Analyze these frames and specify the surgical phase and step being performed."
    ]
    
    general_questions = [
        "Describe what you see in this surgical video.",
        "Provide a detailed description of this surgical procedure.",
        "What is visible in these surgical images?",
        "Give a comprehensive description of the surgical scene.",
        "What details can you observe in these surgical frames?"
    ]
    
    # Convert image paths from GraSP/train/frames/... to /scratch/e0957602/BN4101/grasp_frames/...
    def convert_image_path(image_path):
        """Convert image path to HPC scratch space path."""
        if image_path.startswith("GraSP/train/frames/"):
            # Remove the prefix and add the HPC path
            relative_path = image_path.replace("GraSP/train/frames/", "")
            return f"/scratch/e0957602/BN4101/grasp_frames/{relative_path}"
        return image_path
    
    for sample in grasp_data:
        # Pick a random conversation variant (since you have 5 variations)
        conversation_text = random.choice(sample['conversations']) if sample['conversations'] else ""
        
        # Convert image paths to HPC scratch space
        converted_images = [convert_image_path(img) for img in sample['images']]
        
        # Use tuple of sorted image paths as the group key to identify frame sequences
        # This ensures samples with the same images are grouped together
        sequence_key = tuple(sorted(converted_images))
        
        group_samples = []
        
        # Create LLaMA-Factory format for phase/step identification
        if conversation_text:
            # Add <image> tokens matching the number of images
            num_images = len(converted_images)
            image_tokens = "<image>" * num_images
            user_content = f"{image_tokens}\n{random.choice(phase_step_questions)}"
            
            converted_sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": conversation_text
                    }
                ],
                "images": converted_images
            }
            
            group_samples.append(converted_sample)
        
        # Optional: Add the general response as a second training sample
        # This gives the model more varied training data
        if include_general_response and sample.get('response'):
            # Add <image> tokens matching the number of images
            num_images = len(converted_images)
            image_tokens = "<image>" * num_images
            user_content = f"{image_tokens}\n{random.choice(general_questions)}"
            
            converted_sample_general = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": sample['response']
                    }
                ],
                "images": converted_images
            }
            
            group_samples.append(converted_sample_general)
        
        # Add all samples from this frame sequence to the group
        frame_sequence_groups[sequence_key].extend(group_samples)
    
    # Convert groups to list of (key, samples) tuples for shuffling
    groups_list = list(frame_sequence_groups.items())
    
    # Shuffle groups (not individual samples) to ensure frame sequences stay together
    random.shuffle(groups_list)
    
    # Split groups into train and eval sets
    num_groups = len(groups_list)
    num_train_groups = int(num_groups * train_ratio)
    
    train_groups = groups_list[:num_train_groups]
    eval_groups = groups_list[num_train_groups:]
    
    # Flatten groups into sample lists
    train_samples = []
    for _, samples in train_groups:
        train_samples.extend(samples)
    
    eval_samples = []
    for _, samples in eval_groups:
        eval_samples.extend(samples)
    
    # Create output directories if they don't exist
    for output_file in [train_output_file, eval_output_file]:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    with open(train_output_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    
    # Save evaluation data
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(grasp_data)} input samples into {len(frame_sequence_groups)} frame sequence groups")
    print(f"Training set: {len(train_groups)} groups ({len(train_samples)} samples)")
    print(f"Evaluation set: {len(eval_groups)} groups ({len(eval_samples)} samples)")
    print(f"Saved training data to {train_output_file}")
    print(f"Saved evaluation data to {eval_output_file}")

if __name__ == "__main__":
    # Convert and split dataset into training and evaluation sets
    convert_grasp_to_llamafactory(
        input_file="train/GraSP_caption_InternVL3_5-241B-A28B-Flash_output.json",
        train_output_file="train/grasp_train.json",
        eval_output_file="train/grasp_eval.json",
        include_general_response=True,  # Set to False to only include phase/step samples
        train_ratio=0.8,  # 80% training, 20% evaluation
        random_seed=42  # Fixed seed for reproducibility
    )

