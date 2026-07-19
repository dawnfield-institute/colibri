import json, glob
from tokenizers import Tokenizer
tok = Tokenizer.from_file(glob.glob('/bulk-pool/scratch/hf-cache/hub/models--allenai--OLMoE-1B-7B-0125-Instruct/snapshots/*/tokenizer.json')[0])
text = ('The development of thermodynamics in the nineteenth century began with practical questions about steam engines and ended by reshaping our understanding of nature itself. Sadi Carnot asked how much work could be extracted from heat, and his answer contained the seed of the second law. Clausius sharpened it into entropy, a quantity that never decreases in an isolated system. Boltzmann then connected this macroscopic arrow to the counting of microscopic arrangements, writing entropy as the logarithm of the number of microstates. '
'In software engineering, a cache is judged by its hit rate, but the deeper question is always about the structure of the request stream. A stream with no correlations defeats every policy equally; a stream with locality rewards the policy that models it best. Least-recently-used wins when the recent past predicts the near future, while frequency-based policies win when popularity is stable over long horizons. '
'The recipe calls for two cups of flour, one teaspoon of baking soda, and a pinch of salt. Whisk the dry ingredients together before folding in the wet mixture, taking care not to overwork the batter. Bake at three hundred fifty degrees until a toothpick comes out clean, roughly twenty five minutes depending on the oven.')
ids = tok.encode(text).ids
ref = {'prompt': text[:40], 'prompt_ids': ids[:16], 'full_ids': ids, 'text': '(ppl eval: real mixed-domain text)'}
json.dump(ref, open('ref_ppl.json','w'))
print(f'{len(ids)} tokens total, {len(ids)-16} scored')
