import argparse
from models import APIModelConfig
from utils import my_gpt_completion


parser = argparse.ArgumentParser()


parser.add_argument('--claim')    # path to claim txt
parser.add_argument('--evidence')    # path to evidence document txt
parser.add_argument('--model')  # model name
parser.add_argument('--output')  # output path

args = parser.parse_args()


model = APIModelConfig(model_name=args.model, timeout=120, max_tokens=8192, wait_time=0, api_key=None, temperature=0, source='openai')
with open(f'../prompt/prompt_extract_evidence_statements.txt', 'r') as file:
    instruction = file.read()

with open(args.claim, 'r') as f:
    claim = f.read().strip()

with open(args.evidence, 'r') as f:
    evidence = f.read().strip()

prompt = f'{instruction}\n\n\nTest Input:\n### Claim:\n{claim}\n\n### Document:\n{evidence}\n\nTest Output:'
print(f'-------------------\n{prompt}\n-------------------')


messages = [{"role": "user", "content": prompt}]
response = my_gpt_completion(model.model_name, messages, model.timeout, max_tokens=model.max_tokens, wait_time=model.wait_time, api_key=model.api_key, temperature=model.temperature, source=model.source)
print(response)
print('===================')


if args.output is not None:
    with open(args.output, 'w') as f:
        f.write(response)