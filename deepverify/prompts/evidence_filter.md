You are a claim verification assistant.

Here is a claim:
{CLAIM}

And here are some subclaims that we have decided are entailed by the claim:
{SUBCLAIMS}

Here is a body of evidence extracted from a number of scientific journal articles:
{EVIDENCE}

Our goal is to construct a logical argument for each of the subclaims.  A lot of this evidence is not relevant. Please filter the evidence down to only {TOPK} statements that you think can be used to argue FOR or AGAINST each of the subclaims.

All extractions will be used to prove all subclaims - we do not use different evidence to prove different subclaims.

**IMPORANT: DO NOT EVER OUTPUT MORE THAN {TOPK} PIECES OF EVIDENCE.**

You can think in prose, but at the end of your response please output your filtere evidence in the following format:
```
{{
    "evidence"  : list[str] # list of strings of length {TOPK}
}}
```
    