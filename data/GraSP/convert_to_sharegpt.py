"""
Convert GraSP dataset from conversations/response format to ShareGPT format
for use with LLaMA-Factory and Qwen3-VL.
"""
import json
import os
import random

def convert_grasp_to_llamafactory(input_file, output_file, include_general_response=True):
    """
    Convert GraSP dataset to LLaMA-Factory format with question variations.
    
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
    
    with open(input_file, 'r', encoding='utf-8') as f:
        grasp_data = json.load(f)
    
    llamafactory_data = []
    
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
            
            llamafactory_data.append(converted_sample)
        
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
            
            llamafactory_data.append(converted_sample_general)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(llamafactory_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(grasp_data)} samples to {len(llamafactory_data)} training samples")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Convert training data
    convert_grasp_to_llamafactory(
        input_file="train/GraSP_caption_InternVL3_5-241B-A28B-Flash_output.json",
        output_file="train/GraSP_train_sharegpt.json",
        include_general_response=True  # Set to False to only include phase/step samples
    )

