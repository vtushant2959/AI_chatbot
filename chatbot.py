import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk

# Download the necessary tokenizer for text processing
nltk.download('punkt')

# Load the pre-trained chatbot model (DialoGPT)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chatbot_response(user_input, chat_history=None):
    """
    Function to generate chatbot responses based on user input.
    It keeps track of the chat history for context-aware responses.
    """
    if chat_history is None:
        chat_history = []

    # Tokenize user input
    input_tokens = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Maintain conversation context
    bot_input_ids = torch.cat([torch.tensor(chat_history), input_tokens], dim=-1) if chat_history else input_tokens

    # Generate response from model
    response_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode response to readable text
    response_text = tokenizer.decode(response_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Append the conversation history
    chat_history.append(input_tokens.tolist()[0])  # Append user input
    chat_history.append(response_ids.tolist()[0])  # Append bot response

    return response_text, chat_history

if __name__ == "__main__":
    print("🤖 AI Chatbot: Hello! Type 'exit' to end the chat.")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! 👋")
            break

        response, chat_history = chatbot_response(user_input, chat_history)
        print(f"Chatbot: {response}")
