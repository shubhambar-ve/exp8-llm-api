
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Create the FastAPI application
app = FastAPI(title="LLM Deployment API")

# Load the pretrained model
# This happens once when the application starts
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)


# Define the structure of incoming requests
class PromptRequest(BaseModel):
    prompt: str


# Home endpoint
@app.get("/")
def home():
    return {"message": "LLM API is running"}


# LLM inference endpoint
@app.post("/generate")
def generate(request: PromptRequest):
    result = generator(
        request.prompt,
        max_new_tokens=50
    )

    return {
        "response": result[0]["generated_text"]
    }
