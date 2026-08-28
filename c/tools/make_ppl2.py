import json, glob
from tokenizers import Tokenizer
tok = Tokenizer.from_file(glob.glob('/bulk-pool/scratch/hf-cache/hub/models--allenai--OLMoE-1B-7B-0125-Instruct/snapshots/*/tokenizer.json')[0])
texts = [
 'The measurement of the cosmic microwave background by the COBE, WMAP, and Planck satellites transformed cosmology into a precision science. The angular power spectrum of temperature fluctuations encodes the geometry of the universe, the baryon density, and the amplitude of primordial perturbations. Acoustic oscillations in the photon-baryon plasma before recombination left a characteristic series of peaks whose positions and relative heights constrain the standard cosmological model to percent-level accuracy.',
 'def dijkstra(graph, source):\n    dist = {v: float("inf") for v in graph}\n    dist[source] = 0\n    visited = set()\n    heap = [(0, source)]\n    while heap:\n        d, u = heappop(heap)\n        if u in visited:\n            continue\n        visited.add(u)\n        for v, w in graph[u].items():\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heappush(heap, (dist[v], v))\n    return dist',
 'The Treaty of Westphalia in 1648 ended the Thirty Years War and established principles of state sovereignty that shaped the modern international order. Each state gained the right to determine its own internal affairs, including religion, without external interference. Historians debate how sharply the treaty actually broke with prior practice, but its symbolic role in international relations theory is undeniable.',
 'To prepare the sourdough starter, combine equal weights of flour and water in a clean jar and leave it loosely covered at room temperature. Discard half and feed it daily with fresh flour and water. Within five to seven days, the mixture should double in volume within hours of feeding and smell pleasantly sour, at which point it is ready for baking.',
 'A market is efficient when prices fully reflect available information. The weak form asserts that past prices cannot predict future returns; the semi-strong form extends this to all public information; the strong form includes private information as well. Empirical anomalies such as momentum and post-earnings-announcement drift challenge the hypothesis, though transaction costs complicate their interpretation.',
]
ids = []
for t in texts: ids += tok.encode(t).ids
ref = {'prompt': texts[0][:40], 'prompt_ids': ids[:16], 'full_ids': ids, 'text': '(ppl eval v2: 5-domain, larger n)'}
json.dump(ref, open('/bulk-pool/scratch/colibri/c/ref_ppl2.json','w'))
print(f'{len(ids)} tokens total, {len(ids)-16} scored')
