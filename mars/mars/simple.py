import os
import re
import json
import time
from rich import print as rprint
from .models import Generator, Discriminator
from .utils import extract_final_answer
import sys

def prep_sample(claim, statements, sample_id=0):
    # Create prompt
    formatted_statements = []
    for i, statement in enumerate(statements, 1):
        statement = statement.strip().lower()
        # >>
        # statement = re.sub(r'[^a-z0-9 ,/]', '', statement)  # Remove unnecessary characters
        # --
        statement = statement.encode('utf-8', errors='ignore').decode('utf-8') # [BKJ] Remove non-UTF-8 characters
        # <<
        formatted_statements.append(f"sent{i}: {statement}")
    
    evidence_str    = " ".join(formatted_statements)
    prompt_template = "{EVIDENCE}\n\nclaim: {CLAIM}"
    prompt          = prompt_template.format(EVIDENCE=evidence_str, CLAIM=claim)

    sample  = {'claim_veri_question': prompt, 'claim_veri_answer': None, 'id': sample_id}
    return sample

def run_swap(
        claim           = None, 
        statements      = None,
        output_dir      = None,
        sample          = None,
        sample_id       = 0,
        
        fast_mode       = False,
        num_generations = 3,
        max_steps       = 20,
        num_rollouts    = 32,
        # batch_size_gen  = 24,
        # batch_size_disc = 12,
        beam_width      = 3,
        visualize       = True,
        structure_check = False,
        cmp_per_opt     = 2,
        group_size      = 3,
        API_model       = 'gpt-4o',
        use_thinking_model = False,
        callback        = None
    ):
    print('=' * 40)
    print('sample_id:', sample_id)
    print('claim:', claim)

    if fast_mode:
        print('!!! fast_mode=True !!!')
        # TODO: also change max_steps? That seems to be the main thing 
        num_rollouts = min(8, num_rollouts)

    if use_thinking_model:
        beam_width = num_rollouts

    os.makedirs(output_dir, exist_ok=True)
    
    assert output_dir is not None
    if sample is None:
        sample = prep_sample(claim, statements, sample_id=sample_id)
    else:
        assert claim is None and statements is None
    
    dataset = [sample]

    # Init
    num_gen_dict          = {'output': 1, 'goal': 1, 'state': 1, 'graph': num_generations, 'plan': num_generations, 'action': 1}
    num_future_steps_dict = {             'goal': 0, 'state': 1, 'graph': 0,               'plan': max_steps,       'action': 2, 'final answer': 0}

    print(f'Using API model {API_model} in MARS!')

    agent_gen = Generator(
        # gen_model_id      = None,
        # sem_model_id      = None,
        # model_name        = None,
        # enable_DBM        = False,
        # prob_type         = None,
        use_API           = True,
        API_model         = API_model,
        fast_mode         = fast_mode,
        use_thinking_model = use_thinking_model,
        mixed_act_type    = True,
        allow_assumption  = False
    )
    agent_disc = Discriminator(
        # disc_model_id      = None,
        # model_name         = None,
        use_meta_knwoledge = False,
        # prob_type          = None,
        use_API            = True,
        API_model          = API_model,
        fast_mode          = fast_mode,
        mixed_act_type     = True,
        allow_assumption   = False
    )

    print('num_rollouts:', num_rollouts)

    cnt = 0
    while cnt < max_steps:
        print(f'{"-" * 16} sample {sample_id} step {cnt} {"-" * 16}')
        
        force_termination = False if cnt < max_steps - 1 else True
        
        print('agent_gen.inference: start')
        flag_finish = agent_gen.inference(
            dataset               = dataset,
            output_dir            = output_dir,
            num_rollouts          = num_rollouts,
            # batch_size            = batch_size_gen,
            num_gen_dict          = num_gen_dict,
            num_future_steps_dict = num_future_steps_dict,
            beam_width            = beam_width,
            force_termination     = force_termination, 
            visualize             = visualize
        )
        print('agent_gen.inference: done')

        out = json.load(open(f'{output_dir}/{sample["id"]}.json', 'r'))
        json.dump(out, open(f'{output_dir}/{sample["id"]}-gen-step-{cnt}.json', 'w'))
        
        if callback is not None:
            callback(cnt, out)
        
        if flag_finish:
            break
        
        print('agent_disc.inference: start')
        agent_disc.inference(
            dataset               = dataset,
            output_dir            = output_dir,
            meta_knowledge_path   = None,
            # batch_size            = batch_size_disc,
            cmp_per_opt           = cmp_per_opt,
            group_size            = group_size,
            beam_width            = beam_width,
            num_rollouts          = num_rollouts,
            num_generations       = num_generations,
            structure_check       = structure_check,
            visualize             = visualize
        )
        print('agent_disc.inference: done')

        out = json.load(open(f'{output_dir}/{sample["id"]}.json', 'r'))
        json.dump(out, open(f'{output_dir}/{sample["id"]}-disc-{cnt}.json', 'w'))

        if callback is not None:
            callback(cnt, out)

        cnt += 1
    
    print('Final evaluation with discriminator ...')
    print('agent_disc.inference: start')
    agent_disc.inference(
        dataset             = dataset,
        output_dir          = output_dir,
        meta_knowledge_path = None,
        # batch_size          = batch_size_disc,
        cmp_per_opt         = cmp_per_opt,
        group_size          = group_size,
        beam_width          = 1,
        num_rollouts        = num_rollouts,
        num_generations     = num_generations,
        visualize           = visualize,
        final_agg           = True
    )
    print('agent_disc.inference: done')

    out = json.load(open(f'{output_dir}/{sample["id"]}.json', 'r'))
    json.dump(out, open(f'{output_dir}/{sample["id"]}-disc-final.json', 'w'))

    if callback is not None:
        callback(None, out)

    print(f'sample {sample["id"]} Finished!')



# Change the target sub-claim in the prompt
def _replace_claim_v1(text, new_claim):
    # This pattern captures "claim:" followed by whitespace (group 1),
    # then matches everything until either a newline followed by "### Output:" or the end of the string.
    pattern = r'(claim:\s).*?(?=(?:\n\s*### Output:)|$)'
    return re.sub(pattern, r'\1' + new_claim, text, flags=re.DOTALL)

def _replace_claim_v2(text, new_claim):
    return re.sub(r"the claim '.*?' is 'supported'", f"the claim '{new_claim}' is 'supported'", text)

def _preserve_last_graph(text):
    lines = text.split('\n')
    last_graph_idx = None
    for (i, line) in enumerate(lines):
        if '"Graph ' in line:
            last_graph_idx = i
    
    return '\n'.join(lines[:last_graph_idx+1])

def run_swap_multi_claim(
    claims,
    statements,
    output_dir,
    **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    
    t = time.time()
    
    # For multi-claim verification, we need to verify each sub-claim sequentially
    final_labels = []
    for sample_id, claim in enumerate(claims):
        print('=' * 64, f'sample_id={sample_id:03d} (elapsed={time.time() - t:.2f}s)', '=' * 64)
        if sample_id > 0:
            # prepare the initial graph for the next sub-claim
            sample = prep_sample(claim, statements, sample_id=sample_id)
            
            # modify prompt to include the current sub-claim
            try:
                cur_claim           = sample['claim_veri_question'].split('\n\nclaim:')[-1].strip()
                cur_prompt          = graph['prompt']
                cur_prompt          = _replace_claim_v1(cur_prompt, cur_claim)
                cur_prompt          = _replace_claim_v2(cur_prompt, cur_claim)
                cur_prompt          = _preserve_last_graph(cur_prompt)
                graph['prompt']     = cur_prompt
                sample['rollout']   = {'0': graph}

            except Exception as e:
                rprint(f"[red]!!!!!!!!!!!!! Error preserving last graph: {e} !!!!!!!!!!!!![/red]")
                sample = prep_sample(claim, statements, sample_id=sample_id)
            
            print(f"pre-writing sample {sample['id']} ...")
            with open(f"{output_dir}/{sample['id']}.json", 'w') as f:
                json.dump(sample, f)
        
        elif sample_id == 0:
            sample = prep_sample(claim, statements, sample_id=sample_id)
        
        else:
            raise ValueError(f'sample_id must be greater than 0, but got {sample_id}')
        
        run_swap(output_dir=output_dir, sample=sample, **kwargs)
        
        with open(f"{output_dir}/{sample['id']}.json", 'r') as f:
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

    return final_labels


if __name__ == '__main__':
    
    statements = [
        "If an object or something is in the sunlight, then that object or that something will absorb solar energy.",
        "Daylight is when the sun shines on a location.",
        "The Northern Hemisphere is a kind of hemisphere of Earth.",
        "When the season changes, the amount of daylight will change.",
        "Intensity of sunlight is similar to the amount of sunlight.",
        "As the distance of a location from the North Pole becomes smaller or closer, the amount of daylight received by that location will increase during the summer.",
        "Receiving sunlight is synonymous with absorbing sunlight.",
        "A hemisphere is a part of Earth.",
        "Sunlight means solar energy.",
        "The Earth being tilted on its rotating axis causes seasons.",
        "As the latitude of a location decreases, the amount of sunlight in this location will increase.",
        "Being in the sun is synonymous with being in the sunlight.",
        "Days are a kind of unit for measuring time.",
        "Sunshine means sunlight.",
        "If a place is in summer, then it will have the most sunlight.",
        "Amount of daylight means length of daylight.",
        "Daytime means day.",
        "To receive sunlight means to absorb sunlight.",
        "Hours are a kind of unit for measuring time.",
        "Sunlight shining means sunlight is provided.",
        "A hemisphere of Earth is a kind of place.",
        "Daylight hours means time during which there is daylight.",
        "If places are receiving the same amount of sunlight, then these places will have a similar seasonal weather pattern.",
        "Period of daylight is synonymous with amount of daylight.",
        "Daylight means sunlight.",
    ]

    claim = 'Northern hemisphere will have the most sunlight in summer.'
    run_swap(claim, statements, output_dir='output/sunlight-one')
    
    # claims = [
    #     "Northern hemisphere will have the most sunlight in summer.",
    #     "A location closer to the North Pole receives more daylight in the summer than a location farther from the North Pole.",
    #     "As the latitude of a location decreases, the amount of sunlight in this location will increase.",
    # ]
    # output = run_swap_multi_claim(claims, statements, output_dir='output/sunlight-multi', fast_mode=True, callback=print)
    # breakpoint()
