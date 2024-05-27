# Guide #

## Installation ##

``` 
conda create -n <env> python=3.11
pip install -r requirements.txt
```

## Procedure ##

### Preprocessing ###

- Drop 5 rows of missing messages; this data dont make sense to have sentiments; dropping only 5 rows has no significant impact on the data
- Drop duplicates
- Stratified sampling the data to only take 20% of the data
- Stratified sampling maintains the original representation of the data but with lesser rows 
- The data is then split into training and testing data
- The training data is used for fine-tuning but I was unable to train due to computational limitations
- The testing data is used to evaluate the model, used for all sentiments predictions

### Sentiment-Analysis ###

- Used the pre-trained model BART-Large (Zero-shot classification)
- This model has been trained on large dataset of different types of classes other than just sentiment <br>

- Used the pre-trained model Roberta (Emotion classification)
- This model has been trained on dataset to predict 1 out of the 28 emotions output
- The 8 emotions we are interested in are subset of the 28 emotions from the model
- Relabel to only output 1 out of the 8 emotion output we want


### Summarisation ###

- Used a pre-trained model BART (summary) model to summarise the messages
- Output 2-3 lines however, it doesnt seem to extract the meaningful feature of the summary. 
- Some contexts were lost <br>

- Used llama3-8GB-4bit quantised model already wrapped in the Apple MLX framework for Apple hardware optimisation
- Ease of use as it is already wrapper and just to have to call the function generate()

## Metrics ##

### result ###

- Confusion matrix and classification report for the sentiment analysis models

### summary ###

- summary of the messages
- Can use RAGAS evaluation framework but this will require the ground truth summary of the messages. (time consuming) 

## Optimisation ##

- Transformer models is that the self-attention mechanism grows quadratically in compute and memory with the number of input tokens. This limitation is only magnified in LLMs which handles much longer sequences. To address this, try FlashAttention2 or PyTorch’s scaled dot product attention (SDPA), which are more memory efficient attention implementations and can accelerate inference. (source: HF)

- FlashAttention and FlashAttention-2 break up the attention computation into smaller chunks and reduces the number of intermediate read/write operations to GPU memory to speed up inference. FlashAttention-2 improves on the original FlashAttention algorithm by also parallelizing over sequence length dimension and better partitioning work on the hardware to reduce synchronization and communication overhead. (source: HF)

- Quantisation to lower memory usage; for this we can just select an already quantised version on HF

## Deployment ##

- Selected MLflow for model tracking and deployment
- Open-source platform 
- Tracking of code and results with dashboard
- Packaging format for reproducibility runs on any platform
- Model registry for versioning and deployment of models
- Good wrapper for model deployment.