You are a claim verification assistant.

Here is a claim:
{CLAIM}

And here are some subclaims that we have decided are entailed by the claim:
{SUBCLAIMS}

Here is a piece of evidence extracted from a scientific journal article:
{EVIDENCE}

Please extract axiomatic facts, rules, and findings that are present in the evidence.  If it seems like something that is only relevant in the context of the article, caveat it appropriately.  If it seems like a general scientific truth, phrase it as such.

You can think in prose, but at the end of your response please output your extractions in the following format:
```
{{
    "extractions"  : list[str]
}}
```
    