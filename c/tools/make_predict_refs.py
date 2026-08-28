import json, glob
from tokenizers import Tokenizer
tok = Tokenizer.from_file(glob.glob('/bulk-pool/scratch/hf-cache/hub/models--allenai--OLMoE-1B-7B-0125-Instruct/snapshots/*/tokenizer.json')[0])
N=160
# 4 prompts x 6 domains (held-out prediction test)
P={
 'code':['def binary_search(arr, target):','class LinkedList:\n    def __init__(self):','import numpy as np\ndef normalize(v):','async def fetch_all(urls):'],
 'math':['To prove that the square root of 2 is irrational,','The derivative of a composite function is given by','Consider a group G with identity element e.','The eigenvalues of a symmetric matrix are'],
 'physics':['The Lagrangian of a simple harmonic oscillator is','In quantum mechanics, the expectation value of an operator','Maxwell equations relate the electric and magnetic','The entropy of an ideal gas increases when'],
 'prose':['The old house at the end of the lane','She had never seen the ocean before that','Rain fell steadily through the long afternoon','He remembered the summer they spent'],
 'dialog':['Customer: I need to return this item.','A: Did you finish the report? B: Not yet,','Interviewer: Tell me about your greatest','Doctor: What symptoms have you been'],
 'factual':['The primary function of the mitochondria is','World War II ended in the year','The capital of Australia is','Photosynthesis is the process by which'],
}
for d,prompts in P.items():
    for i,text in enumerate(prompts):
        ids=tok.encode(text).ids
        ref={'prompt':text,'prompt_ids':ids,'full_ids':ids+[0]*N,'text':'(predict test)'}
        json.dump(ref,open(f'ref_pred_{d}_{i}.json','w'))
print('wrote', sum(len(v) for v in P.values()),'refs')
