#!/bin/bash

# Array of random questions
questions=(
    "What is the capital of France?"
    "How does photosynthesis work?"
    "What is the speed of light?"
    "Who wrote 'To Kill a Mockingbird'?"
    "What is the tallest mountain in the world?"
    "How do airplanes fly?"
    "What is the meaning of life?"
    "Who painted the Mona Lisa?"
    "What is quantum mechanics?"
    "What is the largest ocean on Earth?"
    "How does the human immune system work?"
    "What are black holes in space?"
    "Who invented the telephone?"
    "What causes earthquakes?"
    "How do computers store data?"
    "What is artificial intelligence?"
    "What is the theory of relativity?"
    "How does the internet work?"
    "What is the process of evolution?"
    "How do vaccines protect against disease?"
)

# vLLM server endpoint
VLLM_SERVER_URL="http://localhost:8000/v1/comletions"

# Function to send a question to the vLLM server
ping_server() {
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "/public/hf/models/meta-llama/Llama-3.3-70B-Instruct/",
            "prompt": "'"$1"'",
            "max_tokens": 100,
            "temperature": 0.9
        }'
}

# Loop through 10 random questions and send them to the server
for i in {1..20}; do
    random_index=$((RANDOM % ${#questions[@]}))
    echo "Sending question: ${questions[$random_index]}"
    ping_server "${questions[$random_index]}"
    echo "Sleeping for 1 minute before the next request..."
    sleep 60 # Reduced delay between requests
done
