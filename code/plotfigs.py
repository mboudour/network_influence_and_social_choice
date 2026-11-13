import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b",
  "xtick.color":"#52514e","ytick.color":"#52514e","axes.spines.top":False,"axes.spines.right":False,
  "axes.grid":True,"grid.color":"#e6e5e0","grid.linewidth":0.6,"legend.frameon":False,"lines.linewidth":1.6})
BLUE,ORANGE,AQUA,GRAY="#2a78d6","#eb6834","#1baf7a","#8a8984"
SEQ=["#9cc3ef","#4f93e0","#1d5aa8"]   # one hue, light -> dark, for increasing alpha
d=np.load("outputs/figdata12.npz")
pG=0.25-1.5/np.pi*np.arcsin(1/3)
# Fig 1
fig,ax=plt.subplots(figsize=(4.6,3.0))
ax.plot(d["al"],d["pc"],color=BLUE,label=r"limit $p(\alpha)$")
ax.axhline(pG,color=GRAY,ls="--",lw=1.0); ax.text(0.05,pG+0.0018,r"Guilbaud $p_G\approx0.0877$",color="#52514e",fontsize=8)
ax.plot(d["mcA"],d["mcP"],"o",ms=4.5,mfc="white",mec=ORANGE,mew=1.4,label=r"simulation, $k=1000$")
ax.set_xlabel(r"tail index $\alpha$ of the influence weights"); ax.set_ylabel("probability of a Condorcet cycle")
ax.set_xlim(0,2); ax.set_ylim(0,0.095); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("outputs/fig_palpha.pdf"); plt.close(fig)
# Fig 2
fig,ax=plt.subplots(figsize=(4.6,3.0))
ks=d["ks"]; kk=np.logspace(np.log10(5),3,100)
for i,a in enumerate(d["alphas"]):
    mu=a/(a-1); ax.plot(kk,kk*((kk-1)*mu)**(-a),color=SEQ[i],lw=1.2,ls="--")
    ax.plot(ks,d["D"][i],"o",ms=4.5,color=SEQ[i],mec="white",mew=0.8)
    ax.text(1150,d["D"][i][-1],rf"$\alpha={a}$",va="center",fontsize=8,color="#0b0b0b")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(4,2600)
ax.set_xlabel(r"number of committed agents $k$"); ax.set_ylabel("probability of a dictator")
ax.plot([],[],"o",color=GRAY,ms=4.5,label="simulation"); ax.plot([],[],color=GRAY,ls="--",lw=1.2,label=r"$k\,\bar F((k-1)\mu)$")
ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig("outputs/fig_dict.pdf"); plt.close(fig)
# Fig 3
e=np.load("outputs/figdata3.npz"); B,D=e["B"],e["D"]
fig,ax=plt.subplots(figsize=(4.6,3.0))
jit=D*np.exp(rng.normal(0,0.04,len(D))) if (rng:=np.random.default_rng(0)) else D
ax.plot(jit,B,"o",ms=3,color=BLUE,alpha=0.45,mec="none",label=r"$\beta_z$")
x=np.array([1,D.max()*1.2]); ax.plot(x,0.5/(1-0.5)*x,color=GRAY,ls="--",lw=1.0)
ax.text(6.5,14,r"upper bound $\frac{\kappa}{1-\kappa}d_z$",fontsize=8,color="#52514e")
c=np.mean(B[D>=10]/D[D>=10]); ax.plot(x,c*x,color=ORANGE,lw=1.2); ax.text(60,c*35,rf"$\beta_z={c:.3f}\,d_z$",fontsize=8,color="#0b0b0b")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"degree $d_z$"); ax.set_ylabel(r"influence weight $\beta_z$")
fig.tight_layout(); fig.savefig("outputs/fig_weights.pdf"); plt.close(fig)
for f in ("fig_palpha","fig_dict","fig_weights"):
    import subprocess; subprocess.run(["pdftoppm","-r","110","-png","-singlefile",f"outputs/{f}.pdf",f"outputs/{f}"])
# Fig 4: Arrow side p(alpha) vs G-S side q(alpha)
qd=np.load("outputs/qcurve.npz"); d=np.load("outputs/figdata12.npz")
fig,ax=plt.subplots(figsize=(4.8,3.1))
ax.axvspan(1,2,color="#f2f1ec",lw=0); ax.text(1.5,0.003,r"$\gamma=\alpha+1\in(2,3)$",ha="center",fontsize=8,color="#52514e")
ax.plot(d["al"],d["pc"],color=BLUE); ax.text(1.12,0.026,"Condorcet cycle\n(Arrow side)",fontsize=8,color="#0b0b0b")
ax.fill_between(qd["al"],qd["q"]-1.96*qd["se"],qd["q"]+1.96*qd["se"],color=ORANGE,alpha=0.18,lw=0)
ax.plot(qd["al"],qd["q"],color=ORANGE,marker="o",ms=3)
ax.text(0.08,0.092,"manipulable profile\n(Gibbard–Satterthwaite side)",fontsize=8,color="#0b0b0b")
ax.set_xlim(0,2); ax.set_ylim(0,0.115); ax.set_xlabel(r"tail index $\alpha$ of the influence weights"); ax.set_ylabel(r"limiting probability as $k\to\infty$")
fig.tight_layout(); fig.savefig("outputs/fig_pq.pdf"); plt.close(fig)
subprocess.run(["pdftoppm","-r","110","-png","-singlefile","outputs/fig_pq.pdf","outputs/fig_pq"])
# Fig 4 (final): p(alpha); exact mu_k for k=1e3,1e4,1e5; certified LePage bounds for q(alpha)
import json
d=np.load("outputs/figdata12.npz"); pG=0.25-1.5/np.pi*np.arcsin(1/3)
OR=["#f6b48f","#eb6834","#9c3a12"]
fig,ax=plt.subplots(figsize=(4.9,3.2))
ax.axvspan(1,2,color="#f2f1ec",lw=0); ax.text(1.5,0.003,r"$\gamma=\alpha+1\in(2,3)$",ha="center",fontsize=8,color="#52514e")
ax.plot(d["al"],d["pc"],color=BLUE); ax.text(1.12,0.024,"Condorcet cycle, $p(\\alpha)$",fontsize=8,color="#0b0b0b")
for c,k,lab in zip(OR,(1000,10000,100000),("$10^3$","$10^4$","$10^5$")):
    rows=json.load(open(f"outputs/mu_k{k}.json")); a=[r[0] for r in rows]; v=[r[1] for r in rows]
    ax.plot(a,v,color=c,lw=1.3,label=f"manipulable, $k=${lab}")
B=json.load(open("outputs/q_lepage_bounds.json"))
for a,lo,hi in B: ax.plot([a,a],[lo,hi],color="#2b2b2b",lw=1.6)
ax.plot([],[],color="#2b2b2b",lw=1.6,label=r"bounds for the limit $q(\alpha)$")
ax.set_xlim(0,2); ax.set_ylim(0,0.13); ax.set_xlabel(r"tail index $\alpha$ of the weights"); ax.set_ylabel("probability")
ax.legend(loc="upper left",fontsize=7.5)
fig.tight_layout(); fig.savefig("outputs/fig_pq.pdf"); plt.close(fig)
subprocess.run(["pdftoppm","-r","110","-png","-singlefile","outputs/fig_pq.pdf","outputs/fig_pq"])
