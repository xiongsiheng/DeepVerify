import asyncio
import argparse
import dspy
from rich import print as rprint
from fastmcp.client import Client

client = Client("http://localhost:8888/mcp", timeout=240)

class ClaimSignature(dspy.Signature):
    """
    You will be provided with a scientific claim. Use the provided tools to produce a scientific feasibility assessment of this claim.
    Use careful thought, evidentiary reasoning, and scientific rigor in your responses. 
    Bear in-mind that claims can be feasible even if they aren't directly demonstrated in evidence. 
    Produce a narrative argument for your position on the feasibility of the claim.
    Quantify your assessment of the claim's feasibility using a likert score according to this scale:
    | Value | Description |
    | ------| ------------- |
    | -2    | Extremely unlikely to mostly unlikely to be feasible. Significant doubts. |
    | -1    | Somewhat unlikely to be feasible. Moderate doubts but cannot be ruled out. |
    |  0    | Neither unlikely or likely to be feasible. Not enough data and no strong argument for or against. |
    | +1    | Somewhat likely to be feasible. Moderate doubts but can make an argument for why it could be possible. |
    | +2    | Extremely likely to mostly likely to be feasible. Minor doubts to fully confident that this is possible. |
    """
    claim: str = dspy.InputField()
    argument: str = dspy.OutputField()
    likert_score: int = dspy.OutputField()

class HypothesisSignature(dspy.Signature):
    """ 
    You are provided with a scientific reasoning question about a real scientific result. You are tasked with figuring out what novel contributions are required to accomplish the stated results. Your responses should be highlly detailed and capture the key pieces of information that would give an expert attempting to rediscover this result the 'aha moments' that reveal its path to feasibility. Be as specific and detailed as possible; math and specific experimental setup details are highly recommended wherever relevant. It is insufficient to state that something should be done or derived without explaining how it would be done or derived. 
    Some contributions build on other contributions made by the authors in the same paper. In these cases, you will be provided with a list of questions about prior results and the contributions that enabled them to contextualize the current question.
    You may get multiple tries to attempt to answer the provided question. Construct k responses.
    """
    question: str = dspy.InputField(desc="Narrative question about a scientific result.")
    question_context: list[dict] = dspy.InputField(desc="This describes necessary context about the target paper in question-answer format.")
    k: int = dspy.InputField(desc="The number of alternate hypotheses to produce.")
    hypotheses: list[str] = dspy.OutputField(desc="List of your top-k best responses to the posed scientific reasoning question. Each attempt should be independent of the others. Format these as narratives with as much specific detail as possible. Regardless of the number requested, these should each be lengthy. The ground truth that it will be evaluated against is typically on the order of a page in length.")



async def run_claim(**kwargs):
    tool_list = kwargs.pop('tool_list')
    async with client:
        await client.session.initialize()
        tools = await client.list_tools()
        dspy_tools = []
        for tool in tools:
            dspy_tools.append(dspy.Tool.from_mcp_tool(client.session, tool))
        if "all" not in tool_list:
            dspy_tools = [t for t in dspy_tools if t.name in tool_list]
        react = dspy.ReAct(ClaimSignature, tools=dspy_tools)
        return await react.acall(**kwargs)

async def run_hypotheses(**kwargs):
    tool_list = kwargs.pop("tool_list")
    async with client:
        await client.session.initialize()
        tools = await client.list_tools()
        dspy_tools = []
        for tool in tools:
            dspy_tools.append(dspy.Tool.from_mcp_tool(client.session, tool))
        if "all" not in tool_list:
            dspy_tools = [t for t in dspy_tools if t.name in tool_list]
        react = dspy.ReAct(HypothesisSignature, tools=dspy_tools)
        return await react.acall(**kwargs)

def claim_worker(kwargs):
    idx = kwargs.pop('idx')
    return idx, asyncio.run(run_claim(**kwargs))

def hypotheses_worker(kwargs):
    prob_idx = kwargs.pop('problem_idx')
    pid = kwargs.pop('id')
    stage = kwargs.pop('stage')
    return (prob_idx,stage), (pid,stage,asyncio.run(run_hypotheses(**kwargs)))


if __name__ == "__main__":
    import time
    import glob
    import pickle
    import json
    import multiprocessing
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", type=str)
    parser.add_argument("--question", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--tool_list", nargs="*", default=["read_url","search_openscholar","search_arxiv"])
    parser.add_argument("--parallel_factor", type=int, default=8)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--feasibility_assessment", action="store_true")
    parser.add_argument("--path_to_feasibility", action="store_true")
    args = parser.parse_args()

    assert any((args.feasibility_assessment, args.path_to_feasibility)), "Must select one of --feasibility_assessment, --path_to_feasibility"
   
    lm = dspy.LM(args.model_name, temperature=1.0, max_tokens=25000)
    dspy.configure(lm=lm)
    
    if args.feasibility_assessment:
        if args.claim:
            result = asyncio.run(run_claim(tool_list=args.tool_list, claim=args.claim))
            rprint(result['argument'])
            if args.output_path:
                assert args.output_path.endswith("pkl")
                with open(args.output_path, 'wb') as f:
                    pickle.dump({"problems":[{"claim":args.claim}],"results":[result.toDict()]}, f)

        elif args.input_path:
            problems = []
            for path in glob.glob(args.input_path):
                assert path.endswith("jsonl")
                with open(path, 'r') as f:
                    lines = f.readlines()
                problems += list(map(lambda x: json.loads(x), filter(lambda x: x.strip(), lines)))
            print(f"Running {len(problems[:args.max_problems])}/{len(problems)} claims.")
            tick = time.time()
            results = []
            with multiprocessing.Pool(processes=args.parallel_factor) as pool:
                for idx,result in tqdm(pool.imap_unordered(claim_worker, [{"idx":idx, "claim":p["claim"], "tool_list":args.tool_list} for idx,p in enumerate(problems[:args.max_problems])]), total=len(problems[:args.max_problems])):
                    result = result.toDict()
                    results.append((idx, result))
                    if args.output_path:
                        assert args.output_path.endswith("pkl")
                        with open(args.output_path, 'wb') as f:
                            results = sorted(results, key=lambda x: x[0])
                            results_slim = [r[1] for r in results]
                            indices = [r[0] for r in results]
                            problems_slim = [problems[i] for i in indices]
                            pickle.dump({"problems":problems_slim, "results":results_slim}, f)
            tock = time.time()
            print(f"{(tock-tick)//60}s elapsed.")

    elif args.path_to_feasibility:
        if args.question:
            result = asyncio.run(run_hypotheses(tool_list=args.tool_list, question=args.question, question_context=[], k=args.k))
            for hyp in result['hypotheses']:
                rprint(hyp)
            if args.output_path:
                assert args.output_path.endswith("pkl")
                with open(args.output_path, 'wb') as f:
                    pickle.dump({"problems":[{"question":args.question}],"results":[result.toDict()]}, f)

        elif args.input_path:
            problems = []
            for path in glob.glob(args.input_path):
                assert path.endswith("json")
                with open(path, 'r') as f:
                    content = f.read()
                problem = json.loads(content.strip())
                problem["id"] = ".".join(path.split('/')[-1].split(".")[-3:-1])
                problems.append(problem)
            problem_args = []
            for prob_idx, problem in enumerate(problems):
                for i in range(len(problem['stages'])):
                    # TODO: Need handling for knowledge cutoff
                    problem_args.append({"problem_idx":prob_idx,
                                         "id":problem['id'],
                                         "stage":i,
                                         "question":problem['stages'][i]['a_question'],
                                         "k":args.k,
                                         "question_context":problem['stages'][:i],
                                         "tool_list":args.tool_list})
            print(f"Running {len(problem_args[:args.max_problems])}/{len(problem_args)} question stages.")
            tick = time.time()
            results = []
            with multiprocessing.Pool(processes=args.parallel_factor) as pool:
                for idx, (prob_id, stage, result) in tqdm(pool.imap_unordered(hypotheses_worker, problem_args[:args.max_problems]), total=len(problem_args[:args.max_problems])):
                    result = result.toDict()
                    result['problem_id'] = prob_id
                    result['stage'] = stage
                    results.append((idx,result))
                    if args.output_path:
                        assert args.output_path.endswith("pkl")
                        with open(args.output_path, 'wb') as f:
                            results = sorted(results, key=lambda x: x[0])
                            indices = [r[0][0] for r in results]
                            results_slim = list(map(lambda x: x[1], results))
                            problems_slim = [problems[i] for i in indices]
                            pickle.dump({"problems":problems_slim, "results":results_slim}, f)
            tock = time.time()
            print(f"{(tock-tick)//60}s elapsed.")
