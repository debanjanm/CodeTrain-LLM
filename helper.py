def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def create_prompt_squad_v2(context, question, answer):
    try:
        if len(answer) < 1:
            answer = "Cannot Find Answer"
        else:
            answer = answer
    except:
        answer = "Cannot Find Answer"
    prompt_template = f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n{answer}</s>"
    return prompt_template

# def create_prompt(template, validation_rules, **variables):
#     """
#     Create a prompt by replacing variables in the template with their respective values.

#     Parameters:
#     template (str): The template string containing placeholders for variables in the format '{variable_name}'.
#     validation_rules (dict): A dictionary containing variable names as keys and validation functions as values.
#                              The validation function should return True if the variable is valid, otherwise False.
#     variables (dict): A dictionary containing variable names as keys and their corresponding values.

#     Returns:
#     str: The prompt with variables replaced by their values, or a message indicating missing or invalid variables.

#     Raises:
#     KeyError: If any variable in the template is not present in the input 'variables' dictionary.
#     """
#     missing_variables = set(validation_rules.keys()) - set(variables.keys())
#     if missing_variables:
#         missing_variables_msg = f"Variable information not found for: {', '.join(missing_variables)}"
#         return missing_variables_msg

#     for variable_name, validation_func in validation_rules.items():
#         if variable_name not in variables:
#             raise KeyError(f"Variable '{variable_name}' not found in the template.")

#         if not validation_func(variables[variable_name]):
#             return f"Invalid value for variable '{variable_name}'."

#     try:
#         return template.format(**variables)
#     except KeyError as e:
#         raise KeyError(f"Variable '{e.args[0]}' not found in the template.")


# def is_age_valid(age):
#     return isinstance(age, int) and 0 < age < 120

# template = "Hello {name}, today is {day}. You are {age} years old."
# input_variables = {
#     'name': 'John',
#     'day': 'Monday',
#     'age': 150
# }
# validation_rules = {
#     'name': lambda x: isinstance(x, str),
#     'day': lambda x: isinstance(x, str),
#     'age': is_age_valid
# }

# prompt = create_prompt(template, validation_rules, **input_variables)
# print(prompt)




