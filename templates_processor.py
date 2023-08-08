def create_prompt_qa(context, question, answer):
    try:
        if len(answer) < 1:
            answer = "Cannot Find Answer"
        else:
            answer = answer
    except:
        answer = "Cannot Find Answer"
    prompt_template = f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n{answer}</s>"
    return prompt_template

# def remove_last_variable(prompt_template):
#     # Find the last occurrence of '{' and '}' in the prompt template
#     last_open_brace = prompt_template.rfind('{')
#     last_close_brace = prompt_template.rfind('}')
    
#     if last_open_brace != -1 and last_close_brace != -1 and last_close_brace > last_open_brace:
#         # Remove the last variable and all text to the right of it
#         modified_prompt = prompt_template[:last_open_brace]
#         return modified_prompt.strip()  # Remove any extra leading/trailing whitespace
    
#     return prompt_template

# # Example usage
# prompt_template = "### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n{answer}</s>"
# modified_template = remove_last_variable(prompt_template)
# print(modified_template)

