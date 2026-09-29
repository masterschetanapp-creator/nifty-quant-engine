import json, sys, numpy as np, pandas as pd
from nifty_lab import *
DIMS={'L':[500,750],'vol':['garch','gjr'],'anchor':['window','long'],'tails':['gauss','fhs'],
      'drift':['prior','mom','multi'],'k':[0.8,0.9,1.0,1.1,1.2,1.3,1.4]}

def run(lab, spec, start, end, tag):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    o=lab.pos(start-pd.Timedelta(days=1)); o_end=lab.pos(end)
    assert lab.dates[o]<start
    H=o_end-o
    fc=lab.forecast(o,H,spec)                     # forecast uses data<=o only
    ex=lab.score_exam(fc,o_end)                   # <- 'reveal' step
    print(f"\n##### {tag}: model {spec['name']} | origin {lab.dates[o].date()} close {lab.close[o]:.0f} | window ends {lab.dates[o_end].date()} close {lab.close[o_end]:.0f}")
    print("state:",{k:(round(v,2) if isinstance(v,float) else v) for k,v in lab.regime(o).items()})
    print("tilt %.2f%%  median drift %.2f%%  fitted-vol(avg fc) %.1f%%"%(fc['tilt']*100,fc['muH']*100,ex['vol_fc_pct']))
    for k,v in ex.items():
        print(f"   {k}: {[round(x,1) for x in v] if isinstance(v,list) else (round(v,3) if isinstance(v,float) else v)}")
    return o,o_end,H,fc,ex

def correct(lab, spec, o, o_end, H, newname, dims=None):
    DIMS_=dims or DIMS
    R=lab.build_rolling(o,H)
    n=len(R['origins'])
    pin0,cov0=lab.evaluate(R,spec)
    nv=lab.naive_scores(R)
    new,hist=lab.greedy(R,spec,DIMS_)
    new['name']=newname
    pin1,cov1=lab.evaluate(R,new)
    print(f"\n--- CORRECTION STEP ({n} rolling monthly origins {lab.dates[R['origins'][0]].date()} .. {lab.dates[R['origins'][-1]].date()}, ~{n*21/H:.0f} independent 6m windows)")
    print(" incumbent  mean pinball %.4f coverage(50/80/90) %.0f/%.0f/%.0f%%"%(pin0.mean(),cov0['c50']*100,cov0['c80']*100,cov0['c90']*100))
    print(" naive      ",{k:round(v,4) for k,v in nv.items()})
    print(" changes accepted:",hist if hist else "NONE (no candidate passed the >2% & 2-of-3-periods rule)")
    print(" new spec  ",{k:v for k,v in new.items() if k in DIMS or k=='name'})
    print(" new        mean pinball %.4f coverage(50/80/90) %.0f/%.0f/%.0f%%"%(pin1.mean(),cov1['c50']*100,cov1['c80']*100,cov1['c90']*100))
    # full leaderboard sanity: best raw combo (for information only, not adopted blindly)
    best=None
    import itertools
    for combo in itertools.product(*DIMS_.values()):
        c=dict(zip(DIMS_.keys(),combo)); c.update(prior=0.09,lam=10,name='x')
        m=lab.evaluate(R,c)[0].mean()
        if best is None or m<best[0]: best=(m,combo)
    print(" (info) best raw grid combo:",dict(zip(DIMS_.keys(),best[1])),"pinball %.4f"%best[0])
    return new,dict(n=n,hist=hist,pin0=float(pin0.mean()),pin1=float(pin1.mean()),cov0=cov0,cov1=cov1,naive=nv)

if __name__=="__main__":
    lab=Lab()
    V1=json.load(open('spec_v1.json'))
    o,o_end,H,fc,ex=run(lab,V1,'2024-10-29','2025-04-29','WINDOW 1')
    V2,info=correct(lab,V1,o,o_end,H,'v2')
    json.dump(V2,open('spec_v2.json','w'))
