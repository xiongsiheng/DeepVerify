"""
    deepverify.agents.basic
"""

import json
import asyncio
import argparse
import dspy
import random
from typing import Optional, Callable
from rich import print as rprint
from fastmcp.client import Client # FastMCP variant ... could have used langchain_mcp_adapters as well
from deepverify import config
import os

dspy.configure_cache(enable_disk_cache=True,
        enable_memory_cache=True)

def _flatten(ll):
    out = []
    for l in ll:
        if isinstance(l, list):
            out.extend(_flatten(l))
        else:
            out.append(l)
    
    return out

def dedup_evidence(evidence:list) -> list:
    evidence_ids = set()
    evidence_dd = []
    for e in evidence:
        if e['id_raw'] not in evidence_ids:
            evidence_ids.add(e['id_raw'])
            evidence_dd.append(e)
        else:
            continue
    return evidence_dd

class DatabaseQuerySignature(dspy.Signature):
    """
    Given a scientific claim, compose a list of queries for a semantic search database.
    Follow this thought process: claim -> supporting questions -> textual queries.
    You will produce highly detailed and specific queries. 
    These queries should provide good coverage of various ways in which relevant information may appear in literature.
    Your queries will be used to search full-text databases of scientific papers,
    and they should mirror the tone and style of text that might appear in an abstract
    or other body paragraph of a real paper.
    Make your queries roughly a paragraph in length, as greater length and specificity
    improves search quality when using embedding based retrieval.
    """
    k: int = dspy.InputField(desc="The number of queries to generate of each type.")
    claim: str = dspy.InputField(desc="The scientific claim to be decomposed into specific queries.")
    full_text_queries: list[str] = dspy.OutputField(desc="A list of natural language queries to be used in full-text search.")

class EvidenceExtractionSignature(dspy.Signature):
    """
    Given a list of scientific claims and a full-text document, compress the document into a summary of the key methodological details 
    and results that are most relevant to the set of provided claims. Adopt a quick, abbreviated, and factual format for your summary.
    Begin by providing a summary of what the authors actually did, and then follow this with short statements of key results.
    """
    claims: list[str] = dspy.InputField(desc="The scientific claims being studied.")
    full_text: str = dspy.InputField(desc="The full text source.")
    factual_summary: str = dspy.OutputField(desc="A self contained abbreviated summary of the source and its relevant results to the provided claims.")

class EvidenceSummarySignature(dspy.Signature):
    """
    Given a list of scientific claims, and an excerpt from a document, first identify whether the excerpt is specifically relevant to any of the provided claims.
    Adopt a high bar for relevance, so if the document doesn't overlap with the content of a claim in multiple significant ways, then it should be discarded.
    Use the `relevant` parameter to flag whether the document should be retained or not.
    If retained, populate the `summary` field with a summary of the key results described in the excerpt.
    The summary should be written in a general, factual tone, and require no additional context to understand beyond its wording.
    Adopt a general attributive style like 'One paper found that...", "In one paper...", etc. for the summary.
    Estimate the number of provided claims that will likely be directly addressed by the full text of this source. This will be used to prioritize sources for full-text retrieval.
    """
    claims: list[str] = dspy.InputField(desc="The scientific claims being studied.")
    evidence_excerpt: str = dspy.InputField(desc="The evidence excerpt from which to extract statements.")
    relevant: bool = dspy.OutputField(desc="true if the document should be retained, false otherwise.")
    summary: str = dspy.OutputField(desc="A self-contained summary of key results from the evidence excerpt. Empty if document is irrelevant.")
    full_text_metric: int = dspy.OutputField(desc="The number of claims you expect to be directly addressed by some content in the full text of this paper.")

class InitialScientificReasoningSignature(dspy.Signature):
    """
    Given a scientific claim and a set of several full-text sources from literature, outline the beginning of a detailed and rigorous scientific argument for or against the feasibility of the stated claim.
    Be logical and strict in your reasoning. Be pedantic in your interpretation of the 
    exact wording of the claim. We care about whether this specific claim, as-written, is true.
    Combine the provided evidence and any additional information from your own knowledge and begin reasoning about the feasibility of the claim in a logical and structured manner. Use basic in-text citations of the form [evidence id] for the provided evidence and [knowledge] for info from your own knowledge.
    You will be the consumer of this report during the next stage of this pipeline, where you will be provided with many more high-level summaries of key results from other sources. Use this current stage to extract key facts, figures, and other information from this selected set of evidence, and begin ideating about which directions seem most promising for your argument. You will no-longer have access to these full-text sources when you construct your final argument in the next stage.
    """
    claim: str = dspy.InputField(desc="Target claim to prove or disprove the feasibility of.")
    evidence: dict = dspy.InputField(desc="Dictionary of provided full-text evidence.")
    initial_argument: str = dspy.OutputField(desc="Initial thoughts and outline of your future argument with extracted facts and citations.")

class ScientificArgumentSignature(dspy.Signature):
    """
    Given a scientific claim and a set of evidence from literature, construct a detailed and rigorous scientific argument for or against the feasibility of the stated claim.
    Be logical and strict in your reasoning. Be pedantic in your interpretation of the 
    exact wording of the claim. We care about whether this specific claim, as-written, is true.
    Combine the provided evidence and any additional information from your own knowledge and present your work in a logical and structured manner. Use basic in-text citations of the form [evidence id] for the provided evidence and [knowledge] for info from your own knowledge.
    Before beginning to produce an output, go through a period of ideation where you explore various avenues toward proving or disproving feasibility. Pick the most promising one and pursue it in your final response.
    Finally, provide a likert score that numerically summarizes your assessment of the feasibility of the claim:
    | Value | Description |
    | ------| ------------- |
    | -2    | Extremely unlikely to mostly unlikely to be feasible. Significant doubts. |
    | -1    | Somewhat unlikely to be feasible. Moderate doubts but cannot be ruled out. |
    |  0    | Neither unlikely or likely to be feasible. Not enough data and no strong argument for or against. |
    | +1    | Somewhat likely to be feasible. Moderate doubts but can make an argument for why it could be possible. |
    | +2    | Extremely likely to mostly likely to be feasible. Minor doubts to fully confident that this is possible. |
    """
    claim: str = dspy.InputField(desc="Target claim to prove or disprove the feasibility of.")
    evidence: dict = dspy.InputField(desc="Dictionary of provided evidence.")
    argument: str = dspy.OutputField(desc="Narrative argument with citations.")
    feasible: bool = dspy.OutputField(desc="Final label of whether you think the claim is feasible or not based on the constructed argument.")
    likert_score: int = dspy.OutputField(desc="Numerical representation of your judgement of the feasibility of the claim.")

class IterativeScientificArgumentSignature(dspy.Signature):
    """
    Given a scientific claim, the initial outline of an argument, and a set of additional evidence from literature, construct a detailed and rigorous scientific argument for or against the feasibility of the stated claim.
    Be logical and strict in your reasoning. Be pedantic in your interpretation of the 
    exact wording of the claim. We care about whether this specific claim, as-written, is true.
    You may propagate useful findings and corresponding in-text citations from the initial argument, as they were constructed from full-text versions of several sources in your current evidence pool.
    Combine the provided evidence and any additional information from your own knowledge and present your work in a logical and structured manner. Use basic in-text citations of the form [evidence id] for the provided evidence and [knowledge] for info from your own knowledge.
    Before beginning to produce an output, go through a period of ideation where you explore various avenues toward proving or disproving feasibility. Pick the most promising one and pursue it in your final response.
    Finally, provide a likert score that numerically summarizes your assessment of the feasibility of the claim:
    | Value | Description |
    | ------| ------------- |
    | -2    | Extremely unlikely to mostly unlikely to be feasible. Significant doubts. |
    | -1    | Somewhat unlikely to be feasible. Moderate doubts but cannot be ruled out. |
    |  0    | Neither unlikely or likely to be feasible. Not enough data and no strong argument for or against. |
    | +1    | Somewhat likely to be feasible. Moderate doubts but can make an argument for why it could be possible. |
    | +2    | Extremely likely to mostly likely to be feasible. Minor doubts to fully confident that this is possible. |
    """
    claim: str = dspy.InputField(desc="Target claim to prove or disprove the feasibility of.")
    initial_argument: str = dspy.InputField(desc="The beginnings of an argument that you previously constructed after reviewing some full-text sources.")
    evidence: dict = dspy.InputField(desc="Dictionary of provided evidence.")
    argument: str = dspy.OutputField(desc="Narrative argument with citations.")
    feasible: bool = dspy.OutputField(desc="Final label of whether you think the claim is feasible or not based on the constructed argument.")
    likert_score: int = dspy.OutputField(desc="Numerical representation of your judgement of the feasibility of the claim.")

class ExplorativeArgumentSignature(dspy.Signature):
    """
    You will be provided with a scientific claim and asked to produce k arguments for its feasibility, and k arguments against its feasibility.
    Do your best to provide the most-likely arguments for and against the target claim.
    Build each argument by citing specific facts from your own knowledge and following them up with a [knowledge] in-text citation.
    If the target claim were infeasible, then the first infeasible argument you construct should be the most likely reason why- and then the second, etc. The same should go for your feasibility arguments.
    To accomplish this task well, think about the fundamental mechanisms that could cause the claim to be feasible or infeasible, and construct arguments from there.
    """
    claim: str = dspy.InputField(desc="The main claim under evaluation.")
    k:int = dspy.InputField(desc="How many arguments to construct for each position. 2*k in total.")
    arguments: list[str] = dspy.OutputField(desc="Arguments constructed for and against the target claim.")

class SubClaimExtractorSignature(dspy.Signature):
    """
    You will be given a scientific claim, and an initial argument for or against the feasibility of that claim.
    Analyze the argument, and extract the key components of the argument into argumentative sub-claims. Each of these will be used to generate queries for evidence retrieval.
    Each sub-claim should be stated as its own, self-contained, scientific claim about the world. They should all contain enough detail and context that they make sense independent of one another and the original claim+argument pair.
    """
    claim: str = dspy.InputField(desc="The main claim under evaluation.")
    argument: str = dspy.InputField(desc="The argument to be analyzed.")
    sub_claims: list[str] = dspy.OutputField(desc="A list of sub-claims present in the provided argument.")

class RAGBaseline(dspy.Module):
    
    def __init__(self, num_threads:int=8):
        self.client = Client(config.MCP_URL, timeout=3600)
        self.fast_lm = dspy.LM("openai/gpt-5-mini",temperature=1.0,max_tokens=16000)
        self.smart_lm = dspy.LM("openai/gpt-5",temperature=1.0,max_tokens=16000)
        self.query_generator =  dspy.ChainOfThought(DatabaseQuerySignature)
        self.explorative_arguer =  dspy.ChainOfThought(ExplorativeArgumentSignature)
        self.subclaim_extractor =  dspy.ChainOfThought(SubClaimExtractorSignature)
        self.evidence_extractor =  dspy.ChainOfThought(EvidenceExtractionSignature)
        self.evidence_summarizer =  dspy.ChainOfThought(EvidenceSummarySignature)
        self.argument_constructor =  dspy.ChainOfThought(ScientificArgumentSignature)
        self.argument_outliner =  dspy.ChainOfThought(InitialScientificReasoningSignature)
        self.argument_finisher =  dspy.ChainOfThought(IterativeScientificArgumentSignature)
        self._parallel = dspy.Parallel(num_threads=num_threads, disable_progress_bar=True)

    def flatten(self, list_of_lists:list) -> list:
        return [item for sublist in list_of_lists for item in sublist]

    async def parallel(self, func:Callable, lm:dspy.LM, args:list[dict]):
        tasks = [(func,arg_dict) for arg_dict in args]
        with dspy.context(lm=lm):
            results = await dspy.asyncify(self._parallel)(tasks)
        return results
    
    async def ainvoke(self, name, **kwargs):
        async with self.client:
            out = await self.client.call_tool(name, kwargs)
            return json.loads(out.content[0].text)
    
    async def aforward(self, claim:str, document_budget:int=100, fact_budget:int=100, claim_id:str='000'):
        evidence_path = f'evidence/claim_{claim_id}_evidence_new.json'
        os.makedirs('evidence', exist_ok=True)
        collect_evidence = True if not os.path.exists(evidence_path) else False
        load_evidence = False if not os.path.exists(evidence_path) else True
        if collect_evidence:
            # initial arguments for and against
            print("Initial lines of argument...")
            with dspy.context(lm=self.smart_lm):
                results = await dspy.asyncify(self.explorative_arguer)(claim=claim, k=3)

            # generate sub-claims from arguments
            print("Extracting sub-claims...")
            args = [{'claim':claim, 'argument':arg} for arg in results.arguments]
            results = await self.parallel(self.subclaim_extractor, self.fast_lm, args)
            
            # turn sub-claims into database queries
            sub_claims = [claim] + self.flatten(list(map(lambda x: x.sub_claims, results)))
            print(f"{len(sub_claims)} subclaims generated.")
            print("Constructing Queries...")
            args = [{"k":1, "claim":subclaim} for subclaim in sub_claims]
            results = await self.parallel(self.query_generator, self.fast_lm, args)
            full_text_queries = self.flatten(list(map(lambda x: x.full_text_queries, results)))
            print(f"{len(full_text_queries)} full text queries constructed.")

            # Mitigate position bias
            random.shuffle(full_text_queries)
            evidence = []

            # Google Web Search
            print('search_google - start')
            google_batches = await asyncio.gather(
                *[self.ainvoke('search_google', query=query) for query in full_text_queries],
                return_exceptions=True
            )
            google_batches = [gb for gb in google_batches if not isinstance(gb, Exception)]

            def web_to_evidence(item: dict) -> dict:
                url = item.get('url') or ''
                pdf_url = url if url.lower().endswith('.pdf') else None
                return {
                    'text': item.get('content', '') or '',
                    'id_raw': url or item.get('title', ''),  # fallback
                    'metadata': {
                        'title': item.get('title', ''),
                        'url': url,
                        'pdf_url': pdf_url,
                        'source': 'google_web'
                    }
                }

            web_results_flat = []
            for batch in google_batches:
                web_results_flat.extend(batch.get('results', []))
            evidence.extend(web_to_evidence(r) for r in web_results_flat)
            print('search_google - done')

            # Google Scholar
            print('search_google_scholar - start')
            scholar_batches = await asyncio.gather(
                *[self.ainvoke('search_google_scholar', query=query, max_results=10) for query in full_text_queries],
                return_exceptions=True
            )
            scholar_batches = [sb for sb in scholar_batches if not isinstance(sb, Exception)]

            # GoogleScholarResponse format:
            # {
            #   "status": "success"|"error",
            #   "query": str,
            #   "message": str,
            #   "source": "Google Scholar (SerpAPI)",
            #   "num_results": int,
            #   "results": [{
            #       "title": str, "authors": [str], "year": int|None,
            #       "venue": str|None, "citations": int,
            #       "url": str|None, "abstract": str, "source_id": str
            #   }, ...]
            # }

            def scholar_to_evidence(item: dict) -> dict:
                url = item.get('url') or ''
                pdf_url = url if url.lower().endswith('.pdf') else None
                id_raw = url or item.get('source_id') or item.get('title', '')
                return {
                    'text': item.get('abstract', '') or '',
                    'id_raw': id_raw,
                    'metadata': {
                        'title': item.get('title', ''),
                        'url': url,
                        'pdf_url': pdf_url,
                        'year': item.get('year'),
                        'venue': item.get('venue'),
                        'citations': item.get('citations', 0),
                        'authors': item.get('authors', []),
                        'source': 'google_scholar'
                    }
                }

            scholar_results_flat = []
            for batch in scholar_batches:
                if batch.get('status') == 'success':
                    scholar_results_flat.extend(batch.get('results', []))
            evidence.extend(scholar_to_evidence(r) for r in scholar_results_flat)
            print('search_google_scholar - done')

            # Normalize + dedup
            evidence = [json.loads(e) for e in sorted(set([json.dumps(e) for e in evidence]))]
            print(f'{len(evidence)} documents retrieved.')

            # summarize retrieved evidence and filter by relevance
            print('evidence_summary - start')
            evidence = evidence[:document_budget]
            evidence = dedup_evidence(evidence)
            print(f"{len(evidence)} documents after deduplication.")
            args = [{'claims':sub_claims,'evidence_excerpt':e['text']} for e in evidence]
            results = await self.parallel(self.evidence_summarizer, self.fast_lm, args)
            facts2evidence = {}
            facts = []
            to_pull = []
            for i in range(len(evidence)):
                if results[i].relevant:
                    facts2evidence[results[i].summary] = evidence[i]
                    facts.append(results[i].summary)
                    if 'pdf_url' in evidence[i]['metadata'].keys() and evidence[i]['metadata']['pdf_url'] is not None:
                        to_pull.append((results[i].full_text_metric/len(sub_claims),evidence[i]))
                else:
                    continue
            to_pull = sorted(to_pull, key=lambda x: -x[0])
            print(f'{len(facts)} facts extracted.')

            with open(evidence_path, 'w') as f:
                json.dump({'claim':claim, 'facts':facts, 'facts2evidence':facts2evidence}, f)

        if load_evidence:
            with open(evidence_path, 'r') as f:
                data = json.load(f)
                claim = data['claim']
                facts = data['facts'][:fact_budget]
                facts2evidence = data['facts2evidence']

        # Run MARS        
        print('mars - start')
        out = await self.ainvoke('mars', claims=[claim], statements=facts)
        print('mars - done')
        
        for extra_key in ['', '### Input', '### Output']:
            out['claims'][0]['traj'].pop(extra_key)
        likert_score_map = {'very unlikely':-2, 'unlikely':-1, 'neutral':0, 'likely':1, 'very likely':2}
        final_answer = out['claims'][0]['traj']['Final answer'].strip()
        if final_answer.startswith('"') or final_answer.startswith("'"):
            final_answer = final_answer[1:-1]

        likert_score = likert_score_map.get(final_answer, -2)
        
        out = {
            "claim" : claim,
            "argument" : out,
            "likert_score": likert_score,
            "facts2evidence" : facts2evidence
        }
          
        return out



if __name__ == "__main__":
    import time
    import glob
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--parallel_factor", type=int, default=8)
    parser.add_argument("--threads_per_worker", type=int, default=16)
    args = parser.parse_args()
    
    pipeline = RAGBaseline(num_threads=args.threads_per_worker)

    async def loop(tasks) -> list:
        semaphore = asyncio.Semaphore(args.parallel_factor)
        async def worker(**kwargs):
            async with semaphore:
                return await pipeline.aforward(**kwargs)
        coroutines = [worker(**task) for task in tasks]
        results = await asyncio.gather(*coroutines)
        return results
    
    if args.claim:
        result = asyncio.run(pipeline.aforward(claim=args.claim))
        rprint(result['argument'])
        if args.output_path:
            assert args.output_path.endswith("pkl")
            with open(args.output_path, 'wb') as f:
                pickle.dump(result, f)
    elif args.input_path:
        if len(glob.glob(args.input_path)) == 1:
            path = glob.glob(args.input_path)[0]
            assert path.endswith("jsonl")
            with open(path, 'r') as f:
                lines = f.readlines()
            # print(f"{len(lines)} lines found.")

            # lines = lines[:1]
            problems = list(map(lambda x: json.loads(x), filter(lambda x: x.strip(), lines)))
            tick = time.time()
            results = asyncio.run(loop([{'claim':p['claim'], 'claim_id':p['problem_id']} for p in problems]))
            tock = time.time()
            print(f"{(tock-tick)}s elapsed.")
            if args.output_path:
                assert args.output_path.endswith("pkl")
                with open(args.output_path, 'wb') as f:
                    pickle.dump({"problems":problems, "results":results}, f)
        else:
            raise RuntimeError(f"Found multiple matches for input_path: {glob.glob(args.input_path)}")
