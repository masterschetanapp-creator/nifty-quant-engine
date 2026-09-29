import json, pickle, numpy as np, pandas as pd, itertools
from nifty_lab import *
from run_stage import run, correct, DIMS
lab=Lab(); out={}
V1=json.load(open('spec_v1.json'))
NEW=dict(DIMS); NEW['drift']=['prior','mom','multi','bayes']; NEW['volunc']=[0.0,0.25,0.4]

for tag,d in (('W1','2024-10-28'),('W2','2025-10-28'),('W3','2026-09-28')): out['hyp_'+tag]=lab.hypotheses(lab.pos(d))
o1,e1,H1,fc1,ex1=run(lab,V1,'2024-10-29','2025-04-29','WINDOW 1')
V2,info1=correct(lab,V1,o1,e1,H1,'v2')
o2,e2,H2,fc2,ex2=run(lab,V2,'2025-10-29','2026-04-29','WINDOW 2')
V3,info2=correct(lab,V2,o2,e2,H2,'v3')

def pack(fc,o,e,ex):
    cum=fc['cum']; qs=[.05,.10,.25,.50,.75,.90,.95]
    return dict(o=o,e=e,H=fc['H'],P0=float(lab.close[o]),fan=np.quantile(cum,qs,axis=0),act=lab.lp[o+1:e+1]-lab.lp[o],
                dates=lab.dates[o+1:e+1],hist=lab.close[o-125:o+1],hist_dates=lab.dates[o-125:o+1],ex=ex,tilt=fc['tilt'],muH=fc['muH'])
out['W1']=pack(fc1,o1,e1,ex1); out['W2']=pack(fc2,o2,e2,ex2)

# ---------------- FINAL REFRESH (all data through 28/09/2026), extended candidate set ----------------
o_last=len(lab.close)-1; Hr=H2; o_h=o_last-Hr
R=lab.build_rolling(o_h,Hr); n=len(R['origins'])
base=lab.evaluate(R,V3)[0]; nv=lab.naive_scores(R)
print(f"\n=== FINAL REFRESH: {n} monthly origins {lab.dates[R['origins'][0]].date()}..{lab.dates[R['origins'][-1]].date()} (outcomes known by {lab.dates[-1].date()}), ~{n*21/Hr:.0f} independent windows")
pin0,cov0=lab.evaluate(R,V3); print(" incumbent v3: pinball %.4f, coverage 50/80/90 = %.0f/%.0f/%.0f%% | naive zero-drift %.4f, naive prior-drift %.4f"%(pin0.mean(),cov0['c50']*100,cov0['c80']*100,cov0['c90']*100,nv['naive_zero_drift'],nv['naive_prior_drift']))
rows=[]
for dim,vals in NEW.items():
    for v in vals:
        if v==V3.get(dim,{'volunc':0.0}.get(dim)): continue
        cand=dict(V3,**{dim:v}); p,c=lab.evaluate(R,cand)
        th=np.array_split(np.arange(len(p)),3); wins=sum(p[t].mean()<pin0[t].mean() for t in th)
        rows.append((dim,v,round((1-p.mean()/pin0.mean())*100,1),wins,"%.0f/%.0f/%.0f"%(c['c50']*100,c['c80']*100,c['c90']*100)))
rows.sort(key=lambda x:-x[2])
print(" single-change tests vs v3 (gain% in pinball, thirds won of 3, coverage):")
for r in rows[:9]: print("   ",r)
V4,hist4=lab.greedy(R,V3,NEW); V4['name']='v4'
pin4,cov4=lab.evaluate(R,V4)
print(" ACCEPTED CHANGES:",hist4 if hist4 else "NONE")
print(" v4:",{k:V4[k] for k in list(NEW)+['name']}," pinball %.4f coverage %.0f/%.0f/%.0f%%"%(pin4.mean(),cov4['c50']*100,cov4['c80']*100,cov4['c90']*100))
# per-window loss of v4 on the two exam origins (in-sample now, for information)
json.dump(dict(v1=V1,v2=V2,v3=V3,v4=V4),open('specs.json','w'),default=str)
out['refresh']=dict(n=n,pin0=float(pin0.mean()),pin4=float(pin4.mean()),cov0=cov0,cov4=cov4,naive=nv,hist=hist4,rows=rows)

# ---------------- WINDOW 3 forecast ----------------
o3=o_last; origin_date=lab.dates[o3]; s3=pd.Timestamp('2026-10-29'); e3=pd.Timestamp('2027-04-29')
est=lambda a,b: int(round(np.busday_count((a+pd.Timedelta(days=1)).date(),(b+pd.Timedelta(days=1)).date())*0.94))
Hs,He=est(origin_date,s3),est(origin_date,e3); P0=lab.close[o3]
print("\n##### WINDOW 3: origin",origin_date.date(),"close",P0,"| steps to 29/10/2026:",Hs,"to 29/04/2027:",He)
print("state:",{k:(round(v,2) if isinstance(v,float) else v) for k,v in lab.regime(o3).items()})
pct=lambda x:(np.exp(x)-1)*100; qs=[.05,.10,.25,.50,.75,.90,.95]
def summ(spec,label):
    fc=lab.forecast(o3,He,spec); cum=fc['cum']; win=cum[:,He-1]-cum[:,Hs-1]; seg=cum[:,Hs-1:He]-cum[:,[Hs-1]]
    lv=lambda step:[int(P0*np.exp(np.quantile(cum[:,step-1],q))) for q in (.10,.50,.90)]
    print(f" {label:<34} end-lvl(10/50/90): {lv(He)} | start-lvl: {lv(Hs)} | win ret med {pct(np.median(win)):+.1f}% P(up) {(win>0).mean()*100:.0f}% P(<-10%) {(win<np.log(.9)).mean()*100:.0f}% P(touch-10%) {(seg.min(1)<np.log(.9)).mean()*100:.0f}% | avg vol {np.sqrt(fc['hv'].mean()*TD)/100*spec['k']*100:.1f}%")
    return fc,cum,win,seg
fc3,cum,win,seg=summ(V4,"v4 (FINAL)")
summ(V3,"v3 (pre-refresh)")
summ(dict(V4,anchor='long'),"v4 + long-run vol anchor")
summ(dict(V4,volunc=0.4),"v4 + vol-uncertainty 0.4")
summ(dict(V4,drift='bayes'),"v4 + bayes drift")
summ(dict(V4,prior=0.0),"v4 with ZERO drift")
fit=lab.get_fit(o3,V4['vol'],V4['L']); print(" GARCH fit: alpha %.3f beta %.3f gamma %.3f pers %.3f | tilt %.2f%%"%(fit['a'],fit['b'],fit['g'],fit['pers'],fc3['tilt']*100))
fan=np.quantile(cum,qs,axis=0)
print("\nFINAL 6-month window return quantiles [5,10,25,50,75,90,95]%:",[round(float(pct(np.quantile(win,q))),1) for q in qs])
print("FINAL levels at 29/10/2026:",[int(P0*np.exp(fan[i,Hs-1])) for i in range(7)])
print("FINAL levels at 29/04/2027:",[int(P0*np.exp(fan[i,He-1])) for i in range(7)])
print("Prob: up %.0f%% | >+10%% %.0f%% | >+15%% %.0f%% | <-10%% %.0f%% | touch -5%% %.0f%% -10%% %.0f%% -15%% %.0f%%"%((win>0).mean()*100,(win>np.log(1.1)).mean()*100,(win>np.log(1.15)).mean()*100,(win<np.log(.9)).mean()*100,(seg.min(1)<np.log(.95)).mean()*100,(seg.min(1)<np.log(.90)).mean()*100,(seg.min(1)<np.log(.85)).mean()*100))
print("P(end level > 26,336 [1y high]) = %.0f%%"%((P0*np.exp(cum[:,He-1])>26336).mean()*100), "| P(end level < 22,000) = %.0f%%"%((P0*np.exp(cum[:,He-1])<22000).mean()*100))
out['W3']=dict(o=o3,H=He,Hs=Hs,P0=float(P0),fan=fan,origin_date=origin_date,hist=lab.close[o3-200:o3+1],hist_dates=lab.dates[o3-200:o3+1],
               win=win,regime=lab.regime(o3),spec=V4,s3=s3,e3=e3)
out['info1']=info1; out['info2']=info2
pickle.dump(out,open('results.pkl','wb'))
