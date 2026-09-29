import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
out=pickle.load(open('results.pkl','rb'))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25})
NAVY,BLUE,ORG,GRY='#1f3a5f','#4a7fb5','#d9541e','#6b6b6b'

def fan(ax,x,P0,F):
    L=P0*np.exp(F)
    ax.fill_between(x,L[0],L[6],color=BLUE,alpha=.13,lw=0,label='90% band')
    ax.fill_between(x,L[1],L[5],color=BLUE,alpha=.22,lw=0,label='80% band')
    ax.fill_between(x,L[2],L[4],color=BLUE,alpha=.36,lw=0,label='50% band')
    ax.plot(x,L[3],color=NAVY,lw=1.8,label='Model median')

fig,axs=plt.subplots(1,2,figsize=(15,5.6))
for ax,w,title in zip(axs,('W1','W2'),('Window 1: 29 Oct 2024 → 29 Apr 2025  (model v1)','Window 2: 29 Oct 2025 → 29 Apr 2026  (model v2)')):
    d=out[w]; ex=d['ex']; P0=d['P0']
    ax.plot(d['hist_dates'],d['hist'],color=GRY,lw=1.3,label='History (model saw this)')
    fan(ax,d['dates'],P0,d['fan'])
    ax.plot(d['dates'],P0*np.exp(d['act']),color=ORG,lw=2,label='ACTUAL')
    ax.axvline(d['hist_dates'][-1],color='k',ls=':',lw=1)
    t=(f"Actual: {ex['act_ret_pct']:+.1f}%  |  Model median: {ex['fc_median_pct']:+.1f}%  (P(up) = {ex['p_up']*100:.0f}%)\n"
       f"Actual sat at the {ex['pit']*100:.0f}th percentile of the forecast\n"
       f"Vol: forecast {ex['vol_fc_pct']:.1f}% vs actual {ex['vol_act_pct']:.1f}%   |   Worst drawdown {ex['mdd_act_pct']:.1f}%")
    ax.set_xlabel(t,fontsize=9,loc='left',labelpad=8)
    ax.set_title(title,fontsize=11,fontweight='bold',loc='left'); ax.set_ylabel('Nifty 50')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
axs[0].legend(loc='upper left',fontsize=8,ncol=2,frameon=False)
fig.suptitle('Backtest: what the frozen model predicted (using only earlier data) vs what the market did',fontsize=13,fontweight='bold',y=1.0)
fig.tight_layout(); fig.savefig('/mnt/user-data/outputs/nifty_backtest_W1_W2.png',dpi=140,bbox_inches='tight')

# ---------- W3
d=out['W3']; P0=d['P0']; od=pd.Timestamp(d['origin_date']); H=d['H']; Hs=d['Hs']
cal=(d['e3']-od).days; xd=[od+pd.Timedelta(days=j*cal/H) for j in range(1,H+1)]
fig,ax=plt.subplots(figsize=(12.5,6))
ax.plot(d['hist_dates'],d['hist'],color=GRY,lw=1.4,label='History')
fan(ax,xd,P0,d['fan'])
ax.axvline(od,color='k',ls=':',lw=1); ax.axvline(d['s3'],color=ORG,ls='--',lw=1.2); ax.axvline(d['e3'],color=ORG,ls='--',lw=1.2)
ax.axhline(26336,color='#999',ls='--',lw=.8); ax.text(d['hist_dates'][0],26336+80,'1-year high 26,336 (2 Jan 2026)',fontsize=8,color='#777')
F=d['fan']; lv=lambda s,i:P0*np.exp(F[i,s-1])
ax.annotate(f"29 Oct 2026\nmedian {lv(Hs,3):,.0f}\n80%: {lv(Hs,1):,.0f}–{lv(Hs,5):,.0f}",xy=(d['s3'],lv(Hs,3)),xytext=(-185,-85),textcoords='offset points',fontsize=8.5,arrowprops=dict(arrowstyle='-',color=ORG),color=ORG)
ax.annotate(f"29 Apr 2027\nmedian {lv(H,3):,.0f}\n80%: {lv(H,1):,.0f}–{lv(H,5):,.0f}\n90%: {lv(H,0):,.0f}–{lv(H,6):,.0f}",xy=(d['e3'],lv(H,3)),xytext=(-125,55),textcoords='offset points',fontsize=8.5,arrowprops=dict(arrowstyle='-',color=ORG),color=ORG)
ax.set_title('Window 3 forecast (model v4) — last data 28 Sep 2026 at 22,780; orange lines mark the 6-month window',fontsize=11.5,fontweight='bold',loc='left')
ax.set_ylabel('Nifty 50'); ax.legend(loc='upper left',fontsize=8.5,ncol=3,frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
fig.tight_layout(); fig.savefig('/mnt/user-data/outputs/nifty_forecast_W3.png',dpi=140,bbox_inches='tight')
print('saved')
