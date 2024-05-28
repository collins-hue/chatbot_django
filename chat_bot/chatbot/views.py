from django.shortcuts import render
from django.http import JsonResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

def chat_view(request):
    return render(request, 'Templates/chat.html')

def chat(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '')
        
        # Tokenize the input
        input_ids = tokenizer.encode(user_message, return_tensors='pt')
        
        # Generate a response
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return JsonResponse({'message': response})
    return JsonResponse({'message': 'Invalid request method.'})

