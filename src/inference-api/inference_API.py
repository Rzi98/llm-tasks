import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

class SentimentAnalyzer:
    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions"):
        self.model, self.tokenizer, self.config = self.load_model(model_name)
        self.label_mapping = {
            'curiosity': 'Curious to dive deeper',
            'joy': 'Happy',
            'neutral': 'Neutral',
            'surprise': 'Surprised',
            'disgust': 'Disgusted',
            'sadness': 'Sad',
            'fear': 'Fearful',
            'anger': 'Angry'
        }
        self.original_labels = [self.config.id2label[i] for i in range(len(self.config.id2label))]

    def load_model(self, model_name: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, AutoConfig]:
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return model, tokenizer, config

    def get_model_output(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        return logits

    def map_labels(self, logits: torch.Tensor) -> Dict[str, float]:
        probs = F.softmax(logits, dim=-1)
        probs = probs.squeeze().detach().cpu().numpy()
        new_probs = {}
        for orig_label, new_label in self.label_mapping.items():
            if orig_label in self.original_labels:
                new_probs[new_label] = probs[self.original_labels.index(orig_label)]
        return new_probs

    def predict_sentiment(self, text: str) -> Dict[str, float]:
        logits = self.get_model_output(text)
        probs = self.map_labels(logits)
        pred = max(probs, key=probs.get)
        confidence = probs[pred]
        return {"prediction": pred, "confidence": confidence}
    
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    text = "What is the cat doing?"
    result = analyzer.predict_sentiment(text)
    print(result)
    

