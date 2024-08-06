from transformers import pipeline

model_id = "josephj6802/codereviewer"

# Load the model
classifier = pipeline("text-classification", model=model_id)

# Use the model for inference
result = classifier("Your input text here")
print(result)
