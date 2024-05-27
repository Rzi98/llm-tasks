from fastapi import FastAPI, UploadFile, File
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import mlflow
import mlflow.pyfunc

app = FastAPI()

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/llama-7b") ## need to load the model weights locally first
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=["lm_head"]
)
model = AutoModelForCausalLM.from_pretrained("facebook/llama-7b", quantization_config=quantization_config).half().to("cuda")

# Log model with MLflow
class LlamaModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/llama-7b")
        self.model = AutoModelForCausalLM.from_pretrained("facebook/llama-7b", quantization_config=quantization_config).half().to("cuda")

    def predict(self, context, model_input):
        inputs = self.tokenizer(model_input, return_tensors="pt").input_ids.to('cuda')
        with torch.no_grad():
            outputs = self.model(inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
        return self.tokenizer.decode(prediction[0])

mlflow.pyfunc.save_model(
    path="llama_model",
    python_model=LlamaModelWrapper()
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    text = (await file.read()).decode('utf-8')
    inputs = tokenizer(text, return_tensors="pt").input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)
    return {"prediction": tokenizer.decode(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
