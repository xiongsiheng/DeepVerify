import sys
import json
import os
import random
import itertools
from math import ceil
from collections import defaultdict
from tqdm import tqdm

from .utils import *
from .prompt_generation import *

ROOT = os.path.dirname(os.path.abspath(__file__))

class APIModelConfig:
    def __init__(self, model_name='gpt-4o-mini', timeout=120, max_tokens=2048, wait_time=0, api_key=None, temperature=0.7, source='openai'):
        self.model_name = model_name if model_name else 'gpt-4o-mini'
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.wait_time = wait_time
        self.api_key = api_key
        self.temperature = temperature
        self.source = source



class Generator:
    def __init__(self, show_prompt_only=False, use_API=False, API_model=None, fast_mode=False, use_thinking_model=False, mixed_act_type=False, allow_assumption=False):
        '''
        Initialize the Generator object.

        Args:
            show_prompt_only (bool): Whether to only show the prompts without generating completions.
            use_API (bool): Whether to use the API.
            API_model (str): The name of the API model to use.
            fast_mode (bool): Whether to use the fast mode.
            mixed_act_type (bool): Whether to use mixed action types.
            allow_assumption (bool): Whether to allow assumptions.

        Returns:
            None
        '''
        self.show_prompt_only = show_prompt_only  # For debugging purposes
        self.use_API = use_API
        self.API_model = API_model
        self.fast_mode = fast_mode
        self.use_thinking_model = use_thinking_model
        self.mixed_act_type = mixed_act_type
        self.allow_assumption = allow_assumption
        if not show_prompt_only:
            self._build_model()


    def _build_model(self):
        '''
        Build the generator model.
        '''
        if self.use_API:
            self.model = APIModelConfig(model_name=self.API_model, timeout=3600, max_tokens=16000, wait_time=0, api_key=None, temperature=0.3, source='openai')
            return
        
        raise NotImplementedError('Non API model is not implemented yet.')


    def _run_one_batch(self, samples, num_future_steps_dict, force_termination, output_dir, visualize):
        '''
        Run one batch of samples through the generator model.
        
        Args:
            samples (List[Dict]): The list of samples to process.
            force_termination (bool): Whether to force termination.
            output_dir (str): The output directory to save the results.
            visualize (bool): Whether to visualize the results.

        Returns:
            None
        '''
        # obtain the flattened prompts and IDs
        prompts = []
        sampleID_rolloutID = []
        for (idx_sample, sample) in enumerate(samples):
            for rollout_id in sample['rollout']:
                rollout = sample['rollout'][rollout_id]
                if rollout['active'] and '"Final answer":' not in sample['rollout'][rollout_id]['prompt']:
                    prompts.extend([rollout['prompt']] * rollout['num_gen'])
                    sampleID_rolloutID.extend([(idx_sample, rollout_id)] * rollout['num_gen']) 
        
        if self.use_API:
            results = []
            futures = []
            with open(os.path.join(ROOT, './prompt/prompt_SWAP_generator_entailment_bank_Task2.txt'), 'r') as file:
                instruction = file.read()
            if self.mixed_act_type:
                with open(os.path.join(ROOT, './prompt/prompt_SWAP_generator_entailment_bank_Task2_chg_act_type.txt'), 'r') as file:
                    instruction_chg_act_type = file.read()
            if self.allow_assumption:
                with open(os.path.join(ROOT, './prompt/prompt_SWAP_generator_entailment_bank_Task2_chg_act_type_with_assumption_more_examples.txt'), 'r') as file:
                    instruction_assumption = file.read()

            for prompt in prompts:
                if self.mixed_act_type:
                    cur_instruction = instruction if random.uniform(0, 1) > 0.5 else instruction_chg_act_type
                else:
                    cur_instruction = instruction
                if self.allow_assumption:
                    cur_instruction = instruction_assumption
                prompt = f'{cur_instruction}\n\n\nTest:\n{prompt}'
                
                # continue
                messages = [{"role": "user", "content": prompt}]
                
                # retry if generation fails
                cnt_gen = 0
                while cnt_gen < 10: 
                    response = my_gpt_completion(self.model.model_name, messages, self.model.timeout, max_tokens=self.model.max_tokens, wait_time=self.model.wait_time, api_key=self.model.api_key, temperature=self.model.temperature, source=self.model.source)
                    cnt_gen += 1
                    
                    
                    # process the response
                    if '### Output:' in response:
                        response = response.split('### Output:')[1]
                    response = response.strip().split('\n')
                    response = [step.strip() for step in response if len(step.strip()) > 0]
                    response = [step.replace('**', '"') if step.startswith('**') else step for step in response]
                    response = [step for step in response if step.startswith('"')]
                    response = [step for step in response if step.split(':')[0] not in prompt.split("### Output:")[-1]]
                    
                    if len(response) > 0:
                        break

                stepName = response[0].split(':')[0].strip()
                for name in num_future_steps_dict:
                    if name in stepName.lower():
                        cur_futureStepNum = num_future_steps_dict[name]
                
                if self.use_thinking_model:
                    result = '\n'.join(response)
                elif self.fast_mode:
                    if 'plan' in stepName.lower():
                        result = response[0]
                        future_start_idx = 1
                    else:
                        result = '\n'.join(response[:3])
                        future_start_idx = 3
                else:
                    result = response[0]
                    future_start_idx = 1

                if (not self.use_thinking_model) and (len(response) > future_start_idx):
                    future = '\n'.join(response[future_start_idx : future_start_idx + cur_futureStepNum])
                else:
                    future = ''
                
                print(result)
                print('-----')

                results.append(result)
                futures.append(future)

                
        else:
            raise NotImplementedError('Non API model is not implemented yet.')
        
        for idx_prompt in range(len(prompts)):            
            idx_sample, rollout_id = sampleID_rolloutID[idx_prompt]
            
            samples[idx_sample]['rollout'][rollout_id]['responses'].append(results[idx_prompt])
            samples[idx_sample]['rollout'][rollout_id]['futures'].append(futures[idx_prompt])

        for sample in samples:                   
            with open(f'{output_dir}/{sample["id"]}.json', 'w') as f:
                json.dump(sample, f)


    def _rollout_init(self, prompt):
        '''
        Initialize the rollout.

        Args:
            prompt (str): The prompt to initialize the rollout with.

        Returns:
            rollout (Dict): The initialized rollout.
        '''
        rollout = {}
        rollout['active'] = True
        rollout['prompt'] = prompt
        rollout['num_gen'] = 1
        rollout['responses'] = []
        rollout['futures'] = []
        rollout['state_search_history'] = []

        return rollout


    def inference(self, dataset, output_dir, num_rollouts, num_gen_dict, num_future_steps_dict, beam_width, force_termination=False, visualize=False):
        '''
        Perform inference on the given dataset.

        Args:
            dataset (List[Dict]): The dataset to perform inference on.
            output_dir (str): The output directory to save the results.
            force_termination (bool): Whether to force termination.
            visualize (bool): Whether to visualize the results.

        Returns:
            flag_finish (bool): Whether the inference is finished.
        '''
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)


        flag_finish = False
        
        num_processed_samples = 0
        samples = []
        # for sample in tqdm(dataset, total=len(dataset)):
        for sample in dataset:
            sample_inpath = f'{output_dir}/{sample["id"]}.json'
            if os.path.exists(sample_inpath):
                print(f'!! loading from existing file {output_dir}/{sample["id"]}.json')
                with open(sample_inpath, 'r') as f:
                    sample = json.load(f)
                
                if 'flag_correct' in sample:
                    continue

            if 'rollout' not in sample:
                sample['rollout'] = {}

            if len(sample['rollout']) == 0:
                question = sample['claim_veri_question']
                Input = convert_element_format('Problem', question, convert_json=True)
                prompt = f'### Input:\n{Input}\n\n### Output: \n'
                for rollout_id in range(min(beam_width, num_rollouts)):
                    sample['rollout'][str(rollout_id)] = self._rollout_init(prompt)

            num_active_rollout = 0
            for rollout_id in range(len(sample['rollout'])):
                rollout_id = str(rollout_id)
                if not sample['rollout'][rollout_id]['active']:
                    continue
                sample['rollout'][rollout_id]['prompt'] = f"{sample['rollout'][rollout_id]['prompt'].strip()}\n"
                
                if self.show_prompt_only:
                    continue

                if '"Final answer":' in sample['rollout'][rollout_id]['prompt']:
                    continue
                
                # determine the number of generations
                lastStep = sample['rollout'][rollout_id]['prompt'].strip().split('\n')[-1].strip()
                lastStepName = lastStep.split(':')[0].strip()
                for name in num_gen_dict:
                    if name in lastStepName.lower():
                        cur_num_gen = num_gen_dict[name]
                        break

                # we have different search mechanisms for action and state
                if 'action' in lastStepName.lower() or 'goal' in lastStepName.lower():
                    # state search
                    sample['rollout'][rollout_id]['num_gen'] = cur_num_gen - len(sample['rollout'][rollout_id]['responses'])  # only generate the remaining
                    sample['rollout'][rollout_id]['num_gen'] = max(0, sample['rollout'][rollout_id]['num_gen'])
                else:
                    # action search
                    sample['rollout'][rollout_id]['num_gen'] = 1   
                    for _ in range(cur_num_gen-1):
                        next_rollout_id = len(sample['rollout'])
                        if next_rollout_id < num_rollouts:
                            sample['rollout'][str(next_rollout_id)] = self._rollout_init(sample['rollout'][rollout_id]['prompt'])
                num_active_rollout += 1

            # if no active rollout or more than beam_width rollouts (we should first perform discrimination), skip the sample
            if num_active_rollout == 0 or num_active_rollout > beam_width:
                continue

            samples.append(sample)
            num_processed_samples += 1

        if len(samples) > 0:
            self._run_one_batch(samples, num_future_steps_dict, force_termination, output_dir, visualize)
        
        if num_processed_samples == 0:
            flag_finish = True
 
        return flag_finish




class Discriminator():
    def __init__(self, use_meta_knwoledge=False, show_prompt_only=False, use_API=False, API_model=None,
                 fast_mode=False, mixed_act_type=False, allow_assumption=False):
        '''
        Initialize the Discriminator object.

        Args:
            use_meta_knwoledge (bool): Whether to use meta-knowledge.
            show_prompt_only (bool): Whether to only show the prompts without generating completions.
            use_API (bool): Whether to use the API.
            API_model (str): The name of the API model to use.
            fast_mode (bool): Whether to use the fast mode.
            mixed_act_type (bool): Whether to use mixed action types.
            allow_assumption (bool): Whether to allow assumptions.

        Returns:
            None
        '''
        self.use_meta_knwoledge = use_meta_knwoledge
        self.show_prompt_only = show_prompt_only  # For debugging purposes
        self.use_API = use_API
        self.API_model = API_model
        self.fast_mode = fast_mode
        self.mixed_act_type = mixed_act_type
        self.allow_assumption = allow_assumption
        if not show_prompt_only:
            self._build_model()


    def _build_model(self):
        '''
        Build the discriminator model.
        '''
        if self.use_API:
            self.model = APIModelConfig(model_name=self.API_model, timeout=3600, max_tokens=16000, wait_time=0, api_key=None, temperature=0, source='openai')
            return
        
        raise NotImplementedError('Non API model is not implemented yet.')


    def _schedule_all_comparisons(self, options, group_size=3):
        """
        Schedules all possible comparisons with up to 3 options.
        Each comparison is between 2 or 3 options.

        Args:
            options (List[Option]): List of options to compare.

        Returns:
            List[List[Option]]: List of comparisons.
        """
        comparisons = []
        
        # Schedule all possible 2-option comparisons
        comparisons.extend(list(itertools.combinations(options, 2)))
        
        if group_size > 2:
            # Schedule all possible 3-option comparisons
            comparisons.extend(list(itertools.combinations(options, 3)))
        
        return [list(comparison) for comparison in comparisons]


    def _schedule_random_comparisons(self, options, cmp_per_opt=3, group_size=3):
        """
        Schedules a random subset of comparisons ensuring each option participates
        in approximately 'cmp_per_opt' comparisons.
        Useful for larger N to limit the number of comparisons.

        Args:
            options (List[Option]): List of options to compare.
            cmp_per_opt (int): Number of comparisons each option should participate in.
            group_size (int): Number of options in each comparison.

        Returns:
            List[List[Option]]: List of comparisons.
        """
        N = len(options)
        target_total_comparisons = ceil((cmp_per_opt * N) / group_size)
        
        if len(options) < group_size:
            # Generate combinations of all available elements
            all_comparisons = list(itertools.combinations(options, len(options)))
        else:
            # Generate all possible group_size-opt comparisons
            all_comparisons = list(itertools.combinations(options, group_size))
        random.shuffle(all_comparisons)
        
        comparisons = []
        participation_count = defaultdict(int)
        
        for comparison in all_comparisons:
            if all(participation_count[option.id] < cmp_per_opt for option in comparison):
                comparisons.append(list(comparison))
                for option in comparison:
                    participation_count[option.id] += 1
                if len(comparisons) >= target_total_comparisons:
                    break
        
        return comparisons


    def _rank_options(self, options):
        """
        Ranks options based on their scores.
        Returns the list of options sorted by score descending.

        Args:
            options (List[Option]): List of options to rank.

        Returns:
            List[Option]: List of options sorted by score descending.
        """
        return sorted(options, key=lambda x: x.score, reverse=True)


    def _prepare_meta_knowledge(meta_knowledge_path, test_q_id, num_references=1):
        '''
        Prepare the meta-knowledge for the given test question ID.

        Args:
            meta_knowledge_path (str): The path to the meta-knowledge.
            test_q_id (str): The test question ID.
            num_references (int): The number of references to include.

        Returns:
            meta_knowledge (str): The meta knowledge for the test question.
        '''
        with open(f'{meta_knowledge_path}/similar_question_ids.json', 'r') as f:
            data = json.load(f)
        similar_filename_ls = data[test_q_id]
        meta_knowledge = ''
        cnt = 0
        for file in similar_filename_ls:
            file_mapped = f'{meta_knowledge_path}/{file}.json'
            if not os.path.exists(file_mapped):
                continue
            with open(file_mapped, 'r') as f:
                data = json.load(f)
            meta_knowledge += '\n\n' + data['Knowledge']
            cnt += 1
            if cnt >= num_references:
                break
        return meta_knowledge.strip()


    def _post_process(self, data, selected_rollout_ids, selected_option, disc_data, output_dir, filename=None, mode='action_plan', final_agg=False):
        '''
        Post-process the generated results.

        Args:
            data (Dict): The data dictionary.
            disc_data (List): The discrimination data.
            output_dir (str): The output directory to save the results.
            filename (str): The filename to save the results.

        Returns:
            data (Dict): The updated data dictionary.
        '''
        if mode == 'action_plan':
            for rollout_id in data['rollout']:
                cur_rollout = data['rollout'][rollout_id]
                if rollout_id not in selected_rollout_ids:
                    cur_rollout['active'] = False
                    continue
                
                if not final_agg:
                    if len(cur_rollout['responses']) > 0:
                        cur_rollout['prompt'] = f"{cur_rollout['prompt'].strip()}\n{cur_rollout['responses'][0]}" 
                    cur_rollout['responses'] = []
                    cur_rollout['futures'] = []

        elif mode == 'state_pred':
            rollout_id = selected_rollout_ids[0]
            cur_rollout = data['rollout'][rollout_id]
            cur_rollout['prompt'] = f"{cur_rollout['prompt'].strip()}\n{selected_option}"
            cur_rollout['responses'] = []
            cur_rollout['futures'] = []
            cur_rollout['state_search_history'].append(disc_data) 

        if filename is not None:
            with open(f'{output_dir}/{filename}', 'w') as f:
                json.dump(data, f)

        return data


    def _reshape_res(self, prompts_ls, result):
        '''
        Reshape the results to the original list.

        Args:
            prompts_ls (List[List[str]]): The list of prompts.
            result (List[str]): The list of results.

        Returns:
            original_dist (List[List[str]]): The reshaped list of results.
        '''
        original_dist = []
        index = 0
        for sublist in prompts_ls:
            length = len(sublist)
            original_dist.append(result[index:index + length])
            index += length
        return original_dist


    def _run_one_batch(self, output_dir, samples, prompts_ls, options_ls, comparisons_ls, filenames, beam_width, visualize, final_agg, mode='action_plan'):
        '''
        Run one batch of samples through the discriminator model.

        Args:
            output_dir (str): The output directory to save the results.
            samples (List[Dict]): The list of samples to process.
            prompts_ls (List[List[str]]): The list of prompts.
            options_ls (List[List[Option]]): The list of options.
            comparisons_ls (List[List[List[Option]]]): The list of comparisons.
            filenames (List[str]): The list of filenames.
            visualize (bool): Whether to visualize the results.

        Returns:
            None
        '''
        flat_prompts_ls = [item for sublist in prompts_ls for item in sublist]

        if self.use_API:
            flat_results = []
            for prompt in flat_prompts_ls:
                messages = [{"role": "user", "content": prompt}]
                response = my_gpt_completion(self.model.model_name, messages, self.model.timeout, max_tokens=self.model.max_tokens, wait_time=self.model.wait_time, api_key=self.model.api_key, temperature=self.model.temperature, source=self.model.source)
                flat_results.append(response)
        else:
            raise NotImplementedError('Non API model is not implemented yet.')
        
        flat_results = [f'{prompt.strip()}\n{res}' for prompt, res in zip(flat_prompts_ls, flat_results)]
        recovered_res = self._reshape_res(prompts_ls, flat_results)

        for i in range(len(samples)):
            disc_res = recovered_res[i]
            comparisons = comparisons_ls[i]
            
            # print(len(disc_res))

            for idx_res in range(len(disc_res)):
                cur_res = disc_res[idx_res]
                print(cur_res)
                print('=====')

                if 'Conclusion' in cur_res:
                    cur_res = cur_res.split('Conclusion')[-1].strip()
                cur_res = cur_res.lower()

                winner = None
                if 'option 1' in cur_res:
                    winner = comparisons[idx_res][0]
                elif 'option 2' in cur_res and len(comparisons[idx_res]) > 1:
                    winner = comparisons[idx_res][1]
                elif 'option 3' in cur_res and len(comparisons[idx_res]) > 2:
                    winner = comparisons[idx_res][2]
                
                if winner is not None:
                    winner.score += 1

            ranked_options = self._rank_options(options_ls[i])
            final_winners = ranked_options[:beam_width]

            if final_agg:
                answer = samples[i]['claim_veri_answer']
                samples[i]['flag_correct'] = self._judge_final_answer(final_winners[0].description, answer) if answer is not None else None
                samples[i]['final_graph'] = extract_final_graph(final_winners[0].description)

            disc_data = [f'{option.description}\n\n{option.future}' for option in options_ls[i]] if mode == 'state_pred' else None
            self._post_process(samples[i], [winner.rollout_id for winner in final_winners], final_winners[0].description, disc_data, output_dir, filenames[i], mode=mode, final_agg=final_agg)

    def _judge_final_answer(self, pred, gt):
        '''
        Judge the final answer.

        Args:
            pred (str): The predicted answer.
            gt (str): The ground truth answer.

        Returns:
            flag_correct (bool): Whether the answer is correct.
        '''
        gt_result = gt  # entailemnt bank
        flag_correct = None
        if '"Final answer":' in pred:
            boxed_result = extract_final_answer(pred)
            flag_correct = boxed_result.lower() == gt_result.lower()  # entailemnt bank
        
        return flag_correct


    def inference(self, dataset, output_dir, meta_knowledge_path, cmp_per_opt, group_size, beam_width, num_rollouts, num_generations, 
                  deduplicate=True, visualize=False, final_agg=False, structure_check=False):
        '''
        Perform inference on the given dataset.

        Args:
            output_dir (str): The output directory to save the results.
            meta_knowledge_path (str): The path to the meta-knowledge.
            cmp_per_opt (int): The number of comparisons per option.
            group_size (int): The number of options in each comparison.
            deduplicate (bool): Whether to deduplicate the responses.
            visualize (bool): Whether to visualize the results.
            final_agg (bool): Whether to perform final aggregation.
            structure_check (bool): Whether to check the structure of the responses.

        Returns:
            None
        '''        
        if self.use_API:
            with open(os.path.join(ROOT, './prompt/prompt_SWAP_discriminator_entailment_bank_Task2.txt'), 'r') as file:
                instruction = file.read().strip()
            if self.mixed_act_type or self.allow_assumption:
                with open(os.path.join(ROOT, './prompt/prompt_SWAP_discriminator_entailment_bank_Task2_mixed_act_type.txt'), 'r') as file:
                    instruction = file.read().strip()
            
        samples = []
        prompts_ls = []
        options_ls = []
        comparisons_ls = []
        filenames = []
        src_files_state_pred = []

        for data in dataset:
            filename = f'{data["id"]}.json'
            
            if not os.path.exists(f'{output_dir}/{filename}'):
                continue
            with open(f'{output_dir}/{filename}', 'r') as f:
                sample = json.load(f)
            
            if 'flag_correct' in sample:
                continue

            for rollout_id in sample['rollout']:
                if not sample['rollout'][rollout_id]['active']:
                    continue
                cur_num_gen = sample['rollout'][rollout_id]['num_gen']
                break

            problem = sample['claim_veri_question']
            
            # Either we compare different rollouts (planning or aggregating) or we compare different responses from the same rollout (state prediction)
            if final_agg or cur_num_gen == 1:
                context = ''
                responses = []
                futures = []
                rollout_ids = []
                for rollout_id in sample['rollout']:
                    cur_rollout = sample['rollout'][rollout_id]
                    if not cur_rollout['active']:
                        continue
                    if not final_agg:
                        cur_context = cur_rollout['prompt'].split('### Output:')[1].strip()
                        cur_context = cur_context.replace('"\n', '",\n')
                        response = cur_rollout['responses'][0] if len(cur_rollout['responses']) > 0 else ''
                        response = f'{cur_context},\n{response}' if len(response) > 0 else cur_context
                        future = cur_rollout['futures'][0] if len(cur_rollout['futures']) > 0 else ''
                        responses.append(response)
                        futures.append(future)
                    else:
                        responses.append(cur_rollout['prompt'].split('### Output:')[1].strip())
                        # if len(cur_rollout['responses']) == 0:
                        #     continue
                        # responses.append(cur_rollout['responses'][0])
                        futures.append([])
                    rollout_ids.append(rollout_id)
            else:
                src_files_state_pred.append(filename)
                continue

            if not final_agg:
                if deduplicate:
                    # Initialize a dictionary to maintain unique responses and corresponding futures
                    unique_responses = {}
                    for response, future, rollout_id in zip(responses, futures, rollout_ids):
                        if response not in unique_responses:
                            unique_responses[response] = (future, rollout_id)

                    # Extract the deduplicated responses and their corresponding futures
                    responses = list(unique_responses.keys())
                    futures = [unique_responses[response][0] for response in responses]
                    rollout_ids = [unique_responses[response][1] for response in responses]

                if structure_check:
                    # Filter out responses that have incorrect structure
                    responses_filtered = []
                    futures_filtered = []
                    rollout_ids_filtered = []
                    for response, future, rollout_id in zip(responses, futures, rollout_ids):
                        lastStep = response.split('\n')[-1]
                        stepName = lastStep.split(':')[0]
                        stepContent = ':'.join(lastStep.split(':')[1:])
                        if 'graph' not in stepName.lower():
                            responses_filtered.append(response)
                            futures_filtered.append(future)
                            rollout_ids_filtered.append(rollout_id)
                        else:
                            try:
                                if check_graph_structure(eval(stepContent)):
                                    responses_filtered.append(response)
                                    futures_filtered.append(future)
                                    rollout_ids_filtered.append(rollout_id)
                            except:
                                pass

                    responses = responses_filtered
                    futures = futures_filtered
                    rollout_ids = rollout_ids_filtered
                
            # responses = responses[:beam_width*num_generations]
            # futures = futures[:beam_width*num_generations]
            # rollout_ids = rollout_ids[:beam_width*num_generations] 

            responses_filtered = []
            futures_filtered = []
            rollout_ids_filtered = []
            for i in range(len(responses)):
                if len(responses[i].strip()) == 0:
                    continue
                responses_filtered.append(responses[i])
                futures_filtered.append(futures[i])
                rollout_ids_filtered.append(rollout_ids[i])
            responses = responses_filtered
            futures = futures_filtered
            rollout_ids = rollout_ids_filtered

            options = [Option(i, responses[i], futures[i], rollout_ids[i]) for i in range(len(responses))]
            random.shuffle(options)
            num_options = len(responses)
            
            if num_options == 0:
                # Empty invalid responses and futures
                if not final_agg:
                    for rollout_id in sample['rollout']:
                        cur_rollout = sample['rollout'][rollout_id]
                        if not cur_rollout['active']:
                            continue
                        cur_rollout['responses'] = []
                        cur_rollout['futures'] = []
                
                continue
            elif num_options <= beam_width:
                if final_agg:
                    answer = sample['claim_veri_answer']
                    sample['flag_correct'] = self._judge_final_answer(responses[0], answer) if answer is not None else None
                    sample['final_graph'] = extract_final_graph(responses[0])
                self._post_process(sample, rollout_ids, None, None, output_dir, filename)
                continue
            
            meta_knowledge = None
            if self.use_meta_knwoledge:
                meta_knowledge = self._prepare_meta_knowledge(meta_knowledge_path, sample['id'])


            comparisons = self._schedule_random_comparisons(options, cmp_per_opt, group_size)
            # print('len(options):', len(options))
            # print('cmp_per_opt:', cmp_per_opt)
            # print('group_size:', group_size)
            # print('len(comparisons):', len(comparisons))

            prompts = [f'{instruction}\n ### Input: \n{prepare_prompt_for_disciminator(problem, context, [option.description for option in cur_batch], [option.future for option in cur_batch], meta_knowledge)}\n ### Output: \n' for cur_batch in comparisons]
            if self.use_API:
                prompts = [f'{instruction}\n\n\nTest:\n ### Input: \n{prepare_prompt_for_disciminator(problem, context, [option.description for option in cur_batch], [option.future for option in cur_batch], meta_knowledge)}\n\n ### Output: \n' for cur_batch in comparisons]

            if self.show_prompt_only:
                continue

            samples.append(sample)
            prompts_ls.append(prompts)
            options_ls.append(options)
            comparisons_ls.append(comparisons)
            filenames.append(filename)
                
        if len(samples) > 0:
            self._run_one_batch(output_dir, samples, prompts_ls, options_ls, comparisons_ls, filenames, beam_width, visualize, final_agg)

        if len(src_files_state_pred) > 0:
            self.state_pred_refinement(num_rollouts, src_files_state_pred, output_dir, deduplicate, structure_check, meta_knowledge_path, instruction, cmp_per_opt, group_size, visualize, final_agg, num_generations)


    def state_pred_refinement(self, num_rollouts, src_files_state_pred, output_dir, deduplicate, structure_check, meta_knowledge_path, instruction, cmp_per_opt, group_size, visualize, final_agg, num_generations):
        # For state prediction
        for rollout_id in range(num_rollouts):
            rollout_id = str(rollout_id)
            samples = []
            prompts_ls = []
            options_ls = []
            comparisons_ls = []
            filenames = []
            for filename in src_files_state_pred:
                with open(f'{output_dir}/{filename}', 'r') as f:
                    sample = json.load(f)
                
                problem = sample['claim_veri_question']
                
                if (rollout_id not in sample['rollout']) or (not sample['rollout'][rollout_id]['active']):
                    continue
                
                context = sample['rollout'][rollout_id]['prompt'].split('### Output:')[1].strip()
                context = context.replace('"\n', '",\n')
                responses = sample['rollout'][rollout_id]['responses']
                futures = sample['rollout'][rollout_id]['futures']

                if deduplicate:
                    # Initialize a dictionary to maintain unique responses and corresponding futures
                    unique_responses = {}
                    for response, future in zip(responses, futures):
                        if response not in unique_responses:
                            unique_responses[response] = future

                    # Extract the deduplicated responses and their corresponding futures
                    responses = list(unique_responses.keys())
                    futures = list(unique_responses.values())

                if structure_check:
                    # Filter out responses that have incorrect structure
                    responses_filtered = []
                    futures_filtered = []
                    for response, future in zip(responses, futures):
                        lastStep = future.split('\n')[-1]
                        stepName = lastStep.split(':')[0]
                        stepContent = ':'.join(lastStep.split(':')[1:])
                        if 'graph' not in stepName.lower():
                            responses_filtered.append(response)
                            futures_filtered.append(future)
                        else:
                            try:
                                if check_graph_structure(eval(stepContent)):
                                    responses_filtered.append(response)
                                    futures_filtered.append(future)
                            except:
                                pass

                    responses = responses_filtered
                    futures = futures_filtered
                
                responses = responses[:num_generations]
                futures = futures[:num_generations]

                options = [Option(i, responses[i], futures[i], rollout_id) for i in range(len(responses))]
                num_options = len(responses)

                if num_options == 0:
                    continue
                elif num_options <= 1:
                    self._post_process(sample, [rollout_id], options[0].description, [f'{option.description}\n\n{option.future}' for option in options], output_dir, filename, mode='state_pred')
                    continue
                
                meta_knowledge = None
                if self.use_meta_knwoledge:
                    meta_knowledge = self._prepare_meta_knowledge(meta_knowledge_path, sample['id'])

                comparisons = self._schedule_random_comparisons(options, cmp_per_opt, group_size)
                prompts = [f'{instruction}\n ### Input: \n{prepare_prompt_for_disciminator(problem, context, [option.description for option in cur_batch], [option.future for option in cur_batch], meta_knowledge)}\n ### Output: \n' for cur_batch in comparisons]
                if self.use_API:
                    prompts = [f'{instruction}\n\n\nTest:\n ### Input: \n{prepare_prompt_for_disciminator(problem, context, [option.description for option in cur_batch], [option.future for option in cur_batch], meta_knowledge)}\n\n ### Output: \n' for cur_batch in comparisons]

                if self.show_prompt_only:
                    continue

                samples.append(sample)
                prompts_ls.append(prompts)
                options_ls.append(options)
                comparisons_ls.append(comparisons)
                filenames.append(filename)
                    
            if len(samples) > 0:
                self._run_one_batch(output_dir, samples, prompts_ls, options_ls, comparisons_ls, filenames, 1, visualize, final_agg, mode='state_pred')





class Option:
    '''
    Class to represent an option in the comparison task for the discriminator model.
    '''
    def __init__(self, option_id: int, description: str, future: str, rollout_id: str):
        '''
        Initialize the Option object.
        
        Args:
            option_id (int): The ID of the option.
            description (str): The description of the option.
            future (str): The future of the option.

        Returns:
            None
        '''
        self.id = option_id
        self.description = description
        self.future = future
        self.rollout_id = rollout_id
        self.score = 0
