from .models import Generator, Discriminator
from datasets import Dataset, load_dataset
from .utils import *
import gc
import sys
import argparse
from functools import partial
import re



parser = argparse.ArgumentParser()


# To use existing datasets
parser.add_argument('--dataset')
parser.add_argument('--subset', default='all')  # the subset of the dataset to use

# Or to use a new sample
parser.add_argument('--claim')    # path to claim txt
parser.add_argument('--evidence') # path to evidence txt
parser.add_argument('--multi_claim', action='store_true')  # whether to verify multiple claims simultaneously
parser.add_argument('--fast_mode', action='store_true') # whether use fast mode for inference

parser.add_argument('--use_meta_knowledge', action='store_true')  # whether use meta-knparsowledge for discriminator
parser.add_argument('--structure_check', action='store_true')  # whether check the structure of the reasoning process
parser.add_argument('--visualize', action='store_true')  # whether visualize the language model output
parser.add_argument('--batch_process', action='store_true') # whether use batch processing for inference

parser.add_argument('--output_dir')  # the output directory for inference results

parser.add_argument('--max_steps', type=int, default=20)  # the maximum number of steps for reasoning
parser.add_argument('--num_rollouts', type=int, default=8)  # the number of rollouts for each problem
parser.add_argument('--num_generations', type=int, default=5)  # the number of generations for each step
parser.add_argument('--cmp_per_opt', type=int, default=1)  # the number of comparisons per option
parser.add_argument('--group_size', type=int, default=3) # the group size for single-time comparison (recommend: 2 or 3)
parser.add_argument('--beam_width', type=int, default=3)


parser.add_argument('--allow_assumption', action='store_true')  # whether allow assumption in the reasoning process


args = parser.parse_args()





def SWAP(sample, output_dir, meta_knowledge_path, max_steps=20, num_rollouts=8, num_generations=5, 
         cmp_per_opt=1, group_size=3, beam_width=3, use_meta_knowledge=True, structure_check=True, 
         visualize=False, with_graph=False, fast_mode=False, mixed_act_type=True, allow_assumption=False, 
         API_model='gpt-4o'):
    '''
    Run the workflow of SWAP.

    Args:
        output_dir (str): The output directory.
        meta_knowledge_path (str): The path to the meta-knowledge.
        max_steps (int): The maximum number of steps for reasoning.
        num_rollouts (int): The number of rollouts for each problem.
        num_generations (int): The number of generations for each step.
        cmp_per_opt (int): The number of comparisons per option.
        group_size (int): The group size for single-time comparison.
        use_meta_knowledge (bool): Whether use meta-knowledge for discriminator.
        structure_check (bool): Whether check the structure of the reasoning process.
        visualize (bool): Whether visualize the language model output.

    Returns:
        None
    '''
    # Format: laststepName: num_generation
    if fast_mode:
        num_gen_dict = {'output': num_generations, 'goal': 1, 'state': 1, 'graph': num_generations, 'plan': num_generations, 'action': 1}
    else:
        num_gen_dict = {'output': 1, 'goal': 1, 'state': 1, 'graph': num_generations, 'plan': num_generations, 'action': 1}
    
    if not with_graph:
        num_future_steps_dict = {'goal': 0, 'state': 0, 'graph': 0, 'plan': max_steps, 'action': 1, 'final answer': 0}
    else:
        if fast_mode:
            num_future_steps_dict = {'goal': 0, 'state': 1, 'graph': 1, 'plan': max_steps, 'action': 0, 'final answer': 0}
        else:
            num_future_steps_dict = {'goal': 0, 'state': 1, 'graph': 0, 'plan': max_steps, 'action': 2, 'final answer': 0}
    
    dataset_test = [sample]
    
    # Initialize Generator and Discriminator
    agent_gen = Generator(use_API=True, API_model=API_model, fast_mode=fast_mode, mixed_act_type=mixed_act_type, allow_assumption=allow_assumption)
    agent_disc = Discriminator(use_meta_knwoledge=use_meta_knowledge, use_API=True, API_model=API_model, fast_mode=fast_mode, mixed_act_type=mixed_act_type, allow_assumption=allow_assumption)

    cnt = 0
    while cnt < max_steps:
        # Generator perform inference
        force_termination = False if cnt < max_steps-1 else True
        flag_finish = agent_gen.inference(dataset_test, output_dir, num_rollouts, num_gen_dict, num_future_steps_dict, beam_width,
                                            force_termination=force_termination, visualize=visualize)
        
        if flag_finish:
            break
        
        # Discriminator perform inference
        agent_disc.inference(dataset_test, output_dir, meta_knowledge_path, cmp_per_opt, group_size, beam_width, num_rollouts, num_generations,
                             structure_check=structure_check, visualize=visualize)
        
        cnt += 1

    # Perform final aggregation for all rollouts.
    agent_disc.inference(dataset_test, output_dir, meta_knowledge_path, cmp_per_opt, group_size, 1, num_rollouts, num_generations, 
                         visualize=visualize, final_agg=True)
    return


def build_dataset(args):
    '''
    Build the test dataset.

    Args:
        args (argparse.Namespace): The arguments.

    Returns:
        dataset_test: The test dataset.
        meta_knowledge_path: The path to the meta-knowledge.
    '''
    if args.dataset == 'entailment_bank':
        use_ori_data = False
        use_false_data = False
        use_ukn_data = False
        use_OAM_data = True

        if use_ori_data:
            file_path  = '../dataset/entailment_trees/dataset/task_2/test.jsonl'
        elif use_false_data:
            file_path  = '../results/entailment_bank_task2_test_false_hypothesis_by_gpt-4o.jsonl'
        elif use_ukn_data:
            file_path = '../results/entailment_bank_task2_test_not_enough_evidence_hypothesis_manual.jsonl'
        elif use_OAM_data:
            file_path = ['../dataset/200_OAM_Radar_Claims.jsonl', '../dataset/200_OAM_Radar_Claims_missing_evidence.jsonl'][1]
        dataset_filtered = []
        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line)  # Convert JSON string to dictionary
                data['claim_veri_question'] = f"{data['context']}\n\nclaim: {data['hypothesis']}"
                if use_ori_data:
                    data['claim_veri_answer'] = 'Supported'
                elif use_false_data:
                    data['claim_veri_answer'] = 'Refuted'
                elif use_ukn_data:
                    data['claim_veri_answer'] = 'Not Enough Evidence'
                elif use_OAM_data:
                    data['claim_veri_answer'] = data['label']
                dataset_filtered.append(data)

        dataset_test = Dataset.from_list(dataset_filtered)
        meta_knowledge_path = None
    
    return dataset_test, meta_knowledge_path



def swap_wrapper(model, question, extra):
    # Here, question is one element of dataset_test
    return SWAP(
        question,
        extra['output_dir'],
        extra['meta_knowledge_path'],
        max_steps=extra['max_steps'],
        num_rollouts=extra['num_rollouts'],
        num_generations=extra['num_generations'],
        cmp_per_opt=extra['cmp_per_opt'],
        group_size=extra['group_size'],
        beam_width=extra['beam_width'],
        use_meta_knowledge=extra['use_meta_knowledge'],
        structure_check=extra['structure_check'],
        visualize=extra['visualize'],
        allow_assumption=extra['allow_assumption'],
        fast_mode=extra['fast_mode'],
        with_graph=True,
        API_model=model
    )


def process_statements(input_text):
    # Split input text into individual statements
    statements = input_text.strip().split("\n")
    
    # Clean and format each statement
    formatted_statements = []
    for i, statement in enumerate(statements, 1):
        statement = statement.strip().lower()
        statement = re.sub(r'[^a-z0-9 ,/]', '', statement)  # Remove unnecessary characters
        formatted_statements.append(f"sent{i}: {statement}")
    
    return " ".join(formatted_statements)



# Change the target sub-claim in the prompt
def replace_claim_v1(text, new_claim):
    # This pattern captures "claim:" followed by whitespace (group 1),
    # then matches everything until either a newline followed by "### Output:" or the end of the string.
    pattern = r'(claim:\s).*?(?=(?:\n\s*### Output:)|$)'
    return re.sub(pattern, r'\1' + new_claim, text, flags=re.DOTALL)

def replace_claim_v2(text, new_claim):
    return re.sub(r"the claim '.*?' is 'supported'", f"the claim '{new_claim}' is 'supported'", text)

def preserve_last_graph(text):
    lines = text.split('\n')
    last_graph_idx = None
    for (i, line) in enumerate(lines):
        if '"Graph ' in line:
            last_graph_idx = i
    return '\n'.join(lines[:last_graph_idx+1])




if __name__ == '__main__':
    if args.dataset is not None:
        dataset_test, meta_knowledge_path = build_dataset(args)

    if args.claim is not None and args.evidence is not None:
        with open(args.claim, 'r') as f:
            claim = f.read().strip()
        with open(args.evidence, 'r') as f:
            evidence = f.read().strip()
            evidence = process_statements(evidence)

        if args.multi_claim:
            dataset_test = Dataset.from_list([{'claim_veri_question': f"{evidence}\n\nclaim: {sub_claim}", 'claim_veri_answer': None, 'id': 0} for sub_claim in claim.split('\n')])
        else:
            dataset_test = Dataset.from_list([{'claim_veri_question': f"{evidence}\n\nclaim: {claim}", 'claim_veri_answer': None, 'id': 0}])

        meta_knowledge_path = None    


    # Create the extra parameters dictionary
    extra_params = {
        'output_dir': args.output_dir,
        'meta_knowledge_path': meta_knowledge_path,
        'max_steps': args.max_steps,
        'num_rollouts': args.num_rollouts,
        'num_generations': args.num_generations,
        'cmp_per_opt': args.cmp_per_opt,
        'group_size': args.group_size,
        'beam_width': args.beam_width,
        'use_meta_knowledge': args.use_meta_knowledge,
        'structure_check': args.structure_check,
        'visualize': args.visualize,
        'allow_assumption': args.allow_assumption,
        'fast_mode': args.fast_mode
    }

    # Use partial to bind the extra_data argument if desired (or simply pass it via the wrapper)
    swap_partial = partial(swap_wrapper, extra=extra_params)

    if not args.multi_claim:
        if args.batch_process:
            # Now call batch_processing with swap_partial. Note that batch_processing will pass in the model and question.
            batch_processing(
                model='gpt-4o',
                fun=swap_partial,
                question_list=dataset_test,
                num_workers=100,
                timeout_duration=36000
            )
        else:
            # We call swap_wrapper for each claim
            for sample in dataset_test:
                swap_wrapper('gpt-4o', sample, extra_params)
    else:
        # For multi-claim verification, we need to verify each sub-claim sequentially
        final_labels = []
        for (i, sample) in enumerate(dataset_test):
            print(f'======================== sample {i} ========================')
            if i > 0:
                # prepare the initial graph for the next sub-claim
                next_res = {}
                next_res['claim_veri_question'] = sample['claim_veri_question']
                next_res['claim_veri_answer'] = sample['claim_veri_answer']
                next_res['id'] = sample['id']
                
                # Modify the prompt to include the current sub-claim
                cur_claim = sample['claim_veri_question'].split('\n\nclaim:')[-1].strip()
                cur_prompt = graph['prompt']
                cur_prompt = replace_claim_v1(cur_prompt, cur_claim)
                cur_prompt = replace_claim_v2(cur_prompt, cur_claim)
                cur_prompt = preserve_last_graph(cur_prompt)
                graph['prompt'] = cur_prompt

                next_res['rollout'] = {'0': graph}

                with open(f"{args.output_dir}/{sample['id']}.json", 'w') as f:
                    json.dump(next_res, f)

            swap_wrapper('gpt-4o', sample, extra_params)
            
            # Check the label for current sub-claim
            with open(f"{args.output_dir}/{sample['id']}.json", 'r') as f:
                res = json.load(f)
            label = None
            graph = None
            for traj in res['rollout'].values():
                if traj['active']:
                    label = extract_final_answer(traj['prompt'])
                    graph = traj
                    break
            
            final_labels.append(label)
            # if label != 'Supported':
            #     break

        print(final_labels)

import json
from rich import print as rprint
x = json.loads(open('results/bkj_claims/0.json').read())
x = [xx for xx in x['rollout'].values() if xx['active']]
x = x[0]
rprint(x['prompt'])