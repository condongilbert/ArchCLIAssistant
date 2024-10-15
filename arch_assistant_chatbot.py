import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Function to generate bot response
def generate_response(user_input, history_ids=None):
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Concatenate with history
    bot_input_ids = torch.cat([history_ids, new_input_ids], dim=-1) if history_ids is not None else new_input_ids

    # Generate a response
    history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    response = tokenizer.decode(history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response, history_ids

# Static command help dictionary
def get_linux_command_help(user_input):
    command_dict = {
        "install software": "Use pacman to install packages: sudo pacman -S <package-name>",
        "update system": "To update your system, run: sudo pacman -Syu",
        "partition drive": "To partition a drive, use fdisk: sudo fdisk /dev/sdX",
    }
    
    for keyword, help_text in command_dict.items():
        if keyword in user_input.lower():
            return help_text
    return None

# Main chatbot function with command help
def chat_with_command_help():
    print("Arch Linux Assistant Chatbot with Command Help")

    history = None
    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Check for command-specific help first
        command_help = get_linux_command_help(user_input)
        if command_help:
            print(f"Bot: {command_help}")
        else:
            # If no specific command help, use chatbot model
            response, history = generate_response(user_input, history)
            print(f"Bot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chat_with_command_help()