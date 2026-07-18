import json, torch
from transformers import AutoModelForCausalLM
ids = json.load(open('/bulk-pool/scratch/colibri/c/ref_ppl2.json'))['full_ids']
np_ = 16
model = AutoModelForCausalLM.from_pretrained(
    'allenai/OLMoE-1B-7B-0125-Instruct', torch_dtype=torch.bfloat16,
    cache_dir='/bulk-pool/scratch/hf-cache/hub')
model.eval()
x = torch.tensor([ids])
with torch.no_grad():
    logits = model(x).logits[0].float()
lsm = torch.log_softmax(logits, dim=-1)
nll = -sum(lsm[i-1, ids[i]].item() for i in range(np_, len(ids))) / (len(ids)-np_)
print(f'HF bf16 TF-NLL: {nll:.4f} nats/token over {len(ids)-np_} tokens | ppl = {torch.exp(torch.tensor(nll)).item():.2f}')
