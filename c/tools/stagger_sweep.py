import json, threading, time, urllib.request
URL='http://127.0.0.1:8095/v1/chat/completions'
PROMPTS=['Explain the difference between a stack and a queue.','What does big-O notation measure in algorithms?','Explain how binary search works.','What is a race condition in concurrent programming?']
def one(i, out):
    body=json.dumps({'model':'glm-5.2-colibri','messages':[{'role':'user','content':PROMPTS[i%4]}],'max_tokens':48}).encode()
    t0=time.time()
    r=urllib.request.urlopen(urllib.request.Request(URL,body,{'Content-Type':'application/json'}),timeout=3600)
    d=json.loads(r.read()); out[i]=(d['usage']['completion_tokens'], time.time()-t0)
for N in (4,):
    out={}; ths=[threading.Thread(target=one,args=(i,out)) for i in range(N)]
    t0=time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall=time.time()-t0; tot=sum(v[0] for v in out.values())
    print(f'N={N}: {tot} tokens in {wall:.1f}s -> aggregate {tot/wall:.3f} tok/s | per-stream {[f"{v[0]}t/{v[1]:.0f}s" for v in out.values()]}', flush=True)
