# image_feature_extractor
from torchvision import models, transforms
from PIL import Image
import torch
import sys
# text_feature_extractor
import torch
import clip

class ImageFeatureExtractor:
    def __init__(self, model_name="resnet50", device=None):
        """
        init image feature extractor
        :param model_name: "resnet50" or "resnet101"
        :param device: "cuda" or "cpu"
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # choose resnet
        if model_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        elif model_name == "resnet101":
            backbone = models.resnet101(pretrained=True)
            self.feature_dim = 2048
        else:
            raise ValueError("model_name must be 'resnet50' or 'resnet101'")

        # delete the last layer
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval().to(self.device)

        # preprocess
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_path):
        """
        extract image features
        :param image_path: image path
        :return: normalized extracted vector (torch.Tensor)
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(image_tensor).squeeze()  # [2048]
            features = features / features.norm()  # normalization
        return features
    

class TextFeatureExtractor:
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        Initialize the text feature extractor using CLIP.
        :param model_name: CLIP model name, e.g. "ViT-B/32", "ViT-L/14"
        :param device: "cuda" or "cpu"
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = clip.load(model_name, device=self.device)
        self.model.eval()

    def extract(self, text):
        """
        Extract normalized text features from CLIP.
        :param text: input string (or list of strings)
        :return: torch.Tensor with shape [feature_dim] (single text)
                 or [N, feature_dim] (list of N texts)
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text  # assume list of strings

        # Tokenize input text(s)
        tokens = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        # If input was a single string, return [feature_dim] tensor
        if isinstance(text, str):
            return features.squeeze(0)  # shape: [feature_dim]
        else:
            return features  # shape: [N, feature_dim]