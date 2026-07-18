import json, glob
from tokenizers import Tokenizer
tok_path = glob.glob('/bulk-pool/scratch/hf-cache/hub/models--allenai--OLMoE-1B-7B-0125-Instruct/snapshots/*/tokenizer.json')[0]
tok = Tokenizer.from_file(tok_path)
N_NEW = 192
prompts = {
 'prose':   'The lighthouse keeper had not spoken to another human being in four months. When the storm finally broke,',
 'code':    'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n',
 'physics': 'The partition function of a system in thermal equilibrium is defined as the sum over all microstates weighted by',
 'dialog':  'Customer: My order arrived damaged and I would like a refund.\nAgent: I am sorry to hear that. Could you',
 'factual': 'The three primary macronutrients in the human diet are carbohydrates, proteins, and fats. Carbohydrates serve as',
 'math':    'To find the eigenvalues of a 2x2 matrix, we solve the characteristic equation det(A - lambda I) = 0, which expands to',
}
for name, text in prompts.items():
    ids = tok.encode(text).ids
    ref = {'prompt': text, 'prompt_ids': ids, 'full_ids': ids + [0]*N_NEW,
           'text': '(timing/trace only — full_ids padded, match count meaningless)'}
    json.dump(ref, open(f'ref_trace_{name}.json','w'))
    print(f'{name}: {len(ids)} prompt tokens + {N_NEW} new')
