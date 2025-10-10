import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Optional, Tuple, List


class QwenVLFeatureExtractor:
    """
    Feature extractor for Qwen2.5-VL models.
    Extracts hidden state embeddings and optionally generates text outputs.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-VL-7B-Instruct" or "Qwen/Qwen2.5-VL-32B-Instruct")
            device: Device placement strategy ("auto", "cuda", "cpu")
        """
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        
    def _build_prompt_template(
        self, 
        dataset: str, 
        description: str, 
        sample_path: str,
        class_names: List[str]
    ) -> List[dict]:
        """
        Build the prompt message structure with placeholders filled.
        
        Args:
            dataset: Dataset name
            description: Dataset description text
            sample_path: Base path to sample images
            class_names: List of class names
            
        Returns:
            Formatted message list for the model
        """
        intro_text = f"""You are a helpful assistant. Your task is to describe the training conditions of a high-performance federated learning model. The following information is provided in two main parts: first, details about the dataset itself, and second, how this dataset is distributed among the clients for federated training.

**Part 1: Dataset Characteristics**

This section describes the inherent properties and overall structure of the dataset used for training.

**1. Dataset Description**

This model was trained on the {dataset} dataset.

Here is a brief description of the dataset:
{description}

**2. Dataset Categories and Sample Images**

The following are the categories within the dataset, with a sample image for each class.

"""
        
        distribution_text = """**3. Overall Dataset Class Distribution**

The next image is a bar chart that illustrates the distribution of samples across each class in the entire dataset. This chart shows the total number of images available for each class before any client-side distribution."""
        
        client_distribution_text = """---

**Part 2: Client Data Distribution Conditions**

This section details how the dataset, described above, is distributed among the different clients in the federated learning setup, illustrating the non-IID characteristics.

This last image is a bar chart showing how the dataset is distributed among the different clients participating in the federated learning setup. This illustrates the quantity and variety of data that each client holds, highlighting the data heterogeneity."""
        
        summary_request_text = """Based on all the information provided above (dataset description, sample images, class distribution, and client distribution), please provide a summary of the training data conditions for the federated learning model."""
        
        # Build content list with class samples
        content = [{"type": "text", "text": intro_text}]
        
        for class_name in class_names:
            content.append({"type": "text", "text": f"- **Category: {class_name}**"})
            content.append({"type": "image", "image": f"{sample_path}/{class_name}_sample_image.png"})
        
        content.extend([
            {"type": "text", "text": distribution_text},
            {"type": "image", "image": f"{sample_path}/class_distribution.png"},
            {"type": "text", "text": client_distribution_text},
            {"type": "image", "image": f"{sample_path}/client_distribution.png"},
            {"type": "text", "text": summary_request_text}
        ])
        
        return [{"role": "user", "content": content}]
    
    def extract_features(
        self,
        dataset: str,
        description: str,
        sample_path: str,
        class_names: List[str],
        generate: bool = False,
        max_new_tokens: int = 128
    ) -> Tuple[torch.Tensor, Optional[List[str]]]:
        """
        Extract features from the model given dataset information.
        
        Args:
            dataset: Name of the dataset
            description: Brief description of the dataset
            sample_path: Path to sample images directory
            class_names: List of class names in the dataset
            generate: Whether to also generate text output
            max_new_tokens: Maximum number of tokens to generate (if generate=True)
            
        Returns:
            Tuple of (prompt_embedding, generated_text)
            - prompt_embedding: Hidden state vector before the output layer (shape: [1, hidden_size])
            - generated_text: List of generated text strings (None if generate=False)
        """
        # Build messages from template
        messages = self._build_prompt_template(dataset, description, sample_path, class_names)
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Extract hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
        
        # Get last hidden state (embedding before output layer)
        last_hidden_state = outputs.hidden_states[-1]
        prompt_embedding = last_hidden_state[:, -1, :]
        
        # Optionally generate text
        generated_text = None
        if generate:
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                generated_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
        
        return prompt_embedding, generated_text


# Example usage
if __name__ == "__main__":
    # Initialize extractor (can switch to 32B model easily)
    extractor = QwenVLFeatureExtractor(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device="auto"
    )
    
    # Extract features
    embedding, generated = extractor.extract_features(
        dataset="CIFAR-10",
        description="A dataset of 60,000 32x32 color images in 10 classes.",
        sample_path="/path/to/samples",
        class_names=["class1", "class2", "class3", "class4", "class5", 
                     "class6", "class7", "class8", "class9", "class10"],
        generate=True
    )
    
    print(f"Prompt embedding shape: {embedding.shape}")
    print(f"Prompt embedding (first 10 values): {embedding[0, :10]}")
    
    if generated:
        print(f"\nGenerated text:\n{generated}")
