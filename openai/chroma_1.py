# Load the claim dataset
import pandas as pd
import openai
from openai import OpenAI
data_path = '/home/test/src/openai-cookbook/examples/data'

claim_df = pd.read_json(f'{data_path}/scifact_claims.jsonl', lines=True)
print(claim_df.head())

def build_prompt(claim):
    return [
        {"role": "system", "content": "I will ask you to assess a scientific claim. Output only the text 'True' if the claim is true, 'False' if the claim is false, or 'NEE' if there's not enough evidence."},
        {"role": "user", "content": f"""        
Example:

Claim:
0-dimensional biomaterials show inductive properties.

Assessment:
False

Claim:
1/2000 in UK have abnormal PrP positivity.

Assessment:
True

Claim:
Aspirin inhibits the production of PGE2.

Assessment:
False

End of examples. Assess the following claim:

Claim:
{claim}

Assessment:
"""}
    ]

client = OpenAI(base_url="http://127.0.0.1:8000/v1/",api_key="NO-API-KEY")

def assess_claims(claims):
    responses = []
    # Query the OpenAI API
    for claim in claims:
        response = client.completions.create(
            model='gpt-3.5-turbo',
            messages=build_prompt(claim),
            max_tokens=3,
        )
        # Strip any punctuation or whitespace from the response
        responses.append(response.choices[0].message.content.strip('., '))

    return responses

# Let's take a look at 100 claims
samples = claim_df.sample(50)

claims = samples['claim'].tolist() 

def get_groundtruth(evidence):
    groundtruth = []
    for e in evidence:
        # Evidence is empty 
        if len(e) == 0:
            groundtruth.append('NEE')
        else:
            # In this dataset, all evidence for a given claim is consistent, either SUPPORT or CONTRADICT
            if list(e.values())[0][0]['label'] == 'SUPPORT':
                groundtruth.append('True')
            else:
                groundtruth.append('False')
    return groundtruth

evidence = samples['evidence'].tolist()
groundtruth = get_groundtruth(evidence)

def confusion_matrix(inferred, groundtruth):
    assert len(inferred) == len(groundtruth)
    confusion = {
        'True': {'True': 0, 'False': 0, 'NEE': 0},
        'False': {'True': 0, 'False': 0, 'NEE': 0},
        'NEE': {'True': 0, 'False': 0, 'NEE': 0},
    }
    for i, g in zip(inferred, groundtruth):
        confusion[i][g] += 1

    # Pretty print the confusion matrix
    print('\tGroundtruth')
    print('\tTrue\tFalse\tNEE')
    for i in confusion:
        print(i, end='\t')
        for g in confusion[i]:
            print(confusion[i][g], end='\t')
        print()

    return confusion

gpt_inferred = assess_claims(claims)
confusion_matrix(gpt_inferred, groundtruth)