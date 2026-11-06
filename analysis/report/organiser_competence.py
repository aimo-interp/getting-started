"""Organiser-only competence analysis from cached eight-sample 16k originals.

Run: OPENBLAS_NUM_THREADS=1 python analysis/report/organiser_competence.py
All folds group identical problems across models; all fitted probe directions,
gates, and standardisation use training cases only. No new generation is used.
"""
import ast
import json
import sys
from pathlib import Path
from fractions import Fraction
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'cvbench'), str(ROOT / 'solutions/transcript-probe/research')]
import cv
import stack
import train as tp


def reference_answers():
    """Reuse previously checked answers, with explicit exact checks for gaps."""
    tree = ast.parse((ROOT / 'solutions/anchors/thought_anchors/build_cases.py').read_text())
    refs = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'GOLD' for t in n.targets))
    refs = {g: (a, 'existing checked registry: ' + provenance) for g, (a, provenance) in refs.items()}
    # 17017=7*11*13*17; enumerate divisor exponent tuples modulo 12 exactly.
    residues = [0, 1] + [0] * 10
    for prime in (7, 11, 13, 17):
        new = [0] * 12
        for r, count in enumerate(residues):
            for e in range(18):
                new[r * pow(prime, e, 12) % 12] += count
        residues = new
    refs['7ee9d4437899'] = (residues[5], 'exact divisor-residue enumeration, exponents 0..17')
    assert residues[5] == sum(
        pow(7,a,12)*pow(11,b,12)*pow(13,c,12)*pow(17,d,12) % 12 == 5
        for a,b,c,d in product(range(18),repeat=4))
    # Equality of marble probabilities gives 5*b-3*r=-2; n starts 22,30,... .
    ns = [b + (5*b+2)//3 for b in range(7, 100) if (5*b+2) % 3 == 0 and (5*b+2)//3 >= 7]
    refs['8cfb5ed34810'] = (sum(ns[:5]), 'exact urn equation: 5*b-3*r=-2; smallest n=22,30,38,46,54')
    # f(n)-f(n-1)=sigma_3272(n). For M^7, the 2-factor is odd;
    # each of the five odd-prime geometric sums has v2=3 (LTE), hence k=15.
    refs['f02e2f662264'] = (pow(2, 15, 8000), 'divisor-sum identity and LTE: k=5*3=15')
    refs['1c2bec482bb1'] = (sum(4+9*d for d in (1,2,5,10)), 'positive common differences divide gcd(20,30)=10')
    possible = {a+b+a*b for a in range(1,101) for b in range(1,101) if a != b and a+b+a*b <= 100}
    refs['cba4eb5fe5dc'] = (len(possible), 'exhaustive positive distinct a,b <=100')
    # g(k)=f(k-1) is completely additive, with positive integer values at primes.
    # 3521=7*503. Max g(7)=205 from 7^3<=618; max g(503)=617.
    # All choices are feasible with g(other primes)=1 (checked below).
    def omega_except_7(n):
        omega = 0
        for p in range(2,n+1):
            while n % p == 0:
                n //= p
                omega += p != 7
        return omega
    for n in range(2,619):
        m, e7 = n, 0
        while m % 7 == 0:
            m //=7
            e7 +=1
        # Only n=503 involves prime 503 in this range.
        value = 617 if n == 503 else 205*e7+omega_except_7(m)
        assert value <=617
    assert 3521 == 7*503
    refs['45c534741418'] = (821, 'completely additive g: g(7)=1..205, g(503)=1..617; sums 2..822')
    assert 2026 == 2*1013 and all(1013 % d for d in range(2,32))
    refs['b0cb8396e37b'] = (441, 'quadratic in log_2026(x): product roots x=2026^20; divisor count 21^2')
    refs['5e78de5d5cba'] = (277, 'exact travel equations give v=18/5, t=14/5, distance=252/25')
    refs['37172384f0dc'] = (98, 'stationary squared distances u^2+(u^2/2-85)^2 give radii 85 and 13')
    # BD=1628*t, CD=932*t, hence t=k/4, 696/2560<t<1.
    # Tangent circle powers give AE=1628*(1-t^2), AF=932*(1-t^2).
    valid=[]
    for k in range(1,4):
        t=Fraction(k,4)
        if t>Fraction(696,2560) and all((v*(1-t*t)).denominator==1 for v in (1628,932)):
            valid.append(int(2560*t))
    assert valid == [1280]
    refs['11ff3e02ab22'] = (sum(valid), 'angle-bisector and tangent-circle powers; unique t=1/2')
    # Let b=AC, a=AB=BC, q=AI=CI. q^2=a*b^2/(2*a+b).
    # With q/b=t and u=(2*a+b)/b: u/(2*t+1)=125/6,
    # 2*u*t^2=u-1 => (10*t-7)*(50*t^2+60*t+17)=0.
    # Only positive root t=7/10, u=50, so a:b:q=245:10:7.
    refs['d47975ec2ba3'] = (245, 'exact incenter/perimeter equations: minimal integer a:b:q=245:10:7')
    refs['335cfaaf1b6e'] = (588, 'MathArena/aime_2025 problem text match before TikZ; cached answer 588')
    # Independently verify the polygon answer in affine coordinates.
    A=np.array([0,0]); D=np.array([4,0]); E=np.array([20,0]); B=np.array([28,0])
    F=np.array([0,13]); G=np.array([0,65]); C=np.array([0,91]); M=2*F-D; N=2*E-G
    def twice_area(points):
        p=np.array(points); return abs(sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))
    assert Fraction(int(twice_area([A,F,N,B,C,E,M])), int(twice_area([D,E,G,F])))*288 == 588
    return refs


def auc(y,p):
    return roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan


def within_auc(y,p,models):
    # Average only within-configuration robust-fragile pairs, weighted by pairs.
    nums=[]; weights=[]
    for m in np.unique(models):
        k=models==m; w=y[k].sum()*(~y[k]).sum()
        if w:
            nums.append(auc(y[k],p[k])); weights.append(w)
    return np.average(nums,weights=weights) if weights else np.nan


def main():
    cases=cv.load_cases(); cases=cases[cases.source=='train-main-v2'].copy()
    assert len(cases)==80 and cases.group.nunique()==33
    tc={(c['model_id'],c['group']):c for c in tp.load_cases(oss=True)}
    refs=reference_answers(); rows=[]
    for cid,c in cases.iterrows():
        a, provenance=refs[c.group]
        feat=tp.case_features(tc[(c.model_id,c.group)],16384,8)
        answers=feat['answers']; assert len(answers)==8
        screen=tc[(c.model_id,c.group)]['screen']
        good=np.array([x is not None and x==a for x in answers])
        done=np.array([f=='stop' and t<=16384 for f,t in zip(screen['finish'][:8],screen['n_tokens'][:8])])
        rows.append(dict(case_id=cid,model_id=c.model_id,group=c.group,is_robust=bool(c.is_robust),
                         gold=a,gold_source=provenance,correct=int(good.sum()),n=8,
                         solve_rate=good.mean(),finished=int(done.sum()),
                         answers=json.dumps(answers),boxed_count=sum('\\boxed' in t for t in screen['texts'][:8])))
    table=pd.DataFrame(rows).set_index('case_id'); table.to_csv(OUT/'organiser_competence_cases.csv')
    y=cases.is_robust.to_numpy(bool); models=cases.model_id.to_numpy(); groups=cases.group.to_numpy()
    rate=table.solve_rate.to_numpy(); n=len(y)
    print(f'Organiser only: n={n}, problems={len(set(groups))}, robust={y.sum()}, fragile={(~y).sum()}',flush=True)
    print('Rate = correct completed solutions / 8 at 16k; unfinished solutions count as failures.')
    print('All 80 cases have reference answers; provenance saved per case.')
    print(f'Raw solve-rate AUC {auc(y,rate):.3f}; within-model pair AUC {within_auc(y,rate,models):.3f}')
    for m in np.unique(models):
        k=models==m
        print(f'{m}: n={k.sum()} fragile={(~y[k]).sum()} mean solve robust={rate[k&y].mean():.3f} fragile={rate[k&~y].mean():.3f} AUC={auc(y[k],rate[k]):.3f}')
    # Descriptive within-model null, preserving configuration-level label counts.
    # This shuffles cases within configuration, not whole shared problems;
    # group bootstrap below handles repeated problems for effect uncertainty.
    rng=np.random.default_rng(0); obs=within_auc(y,rate,models); null=[]
    for _ in range(5000):
        yp=y.copy()
        for m in np.unique(models):
            k=np.flatnonzero(models==m); yp[k]=rng.permutation(yp[k])
        null.append(within_auc(yp,rate,models))
    print(f'Descriptive one-sided within-model label permutation p={(1+sum(v>=obs for v in null))/5001:.4f} (does not preserve cross-model problem dependence)')
    # Existing OOF predictions: generated rows can be in training, no shared
    # problem crosses train/test. These are descriptive subgroup checks only.
    saved=np.load(OUT/'ablation_candidates.npz',allow_pickle=True); allcases=cv.load_cases()
    idx=allcases.index.get_indexer(cases.index)
    saved_p={k:saved[v][:,idx].mean(0) for k,v in [('stack','attn FULL (tx)'),('tx','tx only (tx)'),('white_box','attn -tx (tx)')]}
    ann=pd.read_csv(ROOT/'analysis/seg_errors/cases_annotated.csv').set_index('case_id')
    saved_p['seg']=ann.reindex(cases.index).p_seg.to_numpy()
    for title,k in [('8/8',rate==1),('<8/8',rate<1),('>=7/8',rate>=.875)]:
        print(f'Subgroup {title}: n={k.sum()} fragile={(~y[k]).sum()}')
        for name,p in saved_p.items():
            print(f'  existing OOF {name}: AUC={auc(y[k],p[k]):.3f}, fragile flagged={((p<.5)&k&~y).sum()}/{(k&~y).sum()}')
    tab=[[int(((rate==1)&~y).sum()),int(((rate==1)&y).sum())],
         [int(((rate<1)&~y).sum()),int(((rate<1)&y).sum())]]
    print(f'Fragility vs 8/8 table {tab}; descriptive Fisher p={fisher_exact(tab).pvalue:.4f}')
    # Load only middle-layer segment means. Gram matrices give exactly the
    # same DIM projections without repeatedly multiplying high-dimensional H.
    H={m:[[],[]] for m in np.unique(models)}
    for c in cases.itertuples():
        s=np.load(cv.DATA/'segs'/cv.safe(c.model_id)/f'{c.group}.npz')['seg_mean'].astype(np.float64)
        layer=round((s.shape[1]-1)/2)
        H[c.model_id][0].append(s[:5,layer].mean(0)); H[c.model_id][1].append(s[5:9,layer].mean(0))
    gram={m:[np.stack(h)@np.stack(h).T for h in hs] for m,hs in H.items()}
    mi={m:np.flatnonzero(models==m) for m in np.unique(models)}
    def dim(tr,site):
        out=np.zeros(n); j=0 if site=='seg_0_5' else 1
        for m,ii in mi.items():
            local=np.flatnonzero(np.isin(ii,tr)); lab=y[ii[local]]
            if not lab.any() or lab.all(): continue
            w=np.zeros(len(ii)); w[local[lab]]=1/lab.sum(); w[local[~lab]]=-1/(~lab).sum()
            K=gram[m][j]; proj=K@w/np.sqrt(max(w@K@w,1e-16)); proj-=proj[local].mean()
            out[ii]=proj/(proj[local].std()+1e-8)
        return out[:,None]
    names=['model','competence','seg','competence+seg','competence_quadratic','competence_quadratic+seg','competence_completion','competence_completion+seg']
    pred={name:np.zeros((20,n)) for name in names}; rawseg=np.zeros((20,n,2))
    onehot=np.column_stack([models==m for m in np.unique(models)]).astype(float)
    for r in range(20):
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=r).split(rate,y,groups):
            seg=stack.safe_seg_scores([None]*n,models,y,groups,tr,scorer=dim)
            rawseg[r,te]=seg[te]
            # Within-configuration training-only normalisation of solve rate.
            z=np.zeros(n); z2=np.zeros(n); finish=np.zeros(n)
            fr=table.finished.to_numpy()/8
            for m,ii in mi.items():
                it=np.intersect1d(ii,tr)
                for raw,out in [(rate,z),(rate**2,z2),(fr,finish)]:
                    sd=raw[it].std(); out[ii]=(raw[ii]-raw[it].mean())/(sd if sd>1e-9 else 1.)
            comp=np.column_stack([onehot,z]); quad=np.column_stack([comp,z2]); completion=np.column_stack([comp,finish])
            Xs={'model':onehot,'competence':comp,'seg':np.column_stack([onehot,seg]),
                'competence+seg':np.column_stack([comp,seg]),'competence_quadratic':quad,
                'competence_quadratic+seg':np.column_stack([quad,seg]),'competence_completion':completion,
                'competence_completion+seg':np.column_stack([completion,seg])}
            for name,X in Xs.items():
                clf=LogisticRegression(C=.1,max_iter=2000).fit(X[tr],y[tr]); pred[name][r,te]=clf.predict_proba(X[te])[:,1]
        print(f'CV repeat {r+1}/20',file=sys.stderr,flush=True)
    print('\nFresh organiser-only training; 20 x 5-fold grouped CV, C=.1, model intercept indicators in every model.')
    for name,p in pred.items():
        print(f'{name:28s} acc={((p>=.5)==y).mean():.3f} AUC={np.mean([auc(y,v) for v in p]):.3f} logloss={np.mean([log_loss(y,v) for v in p]):.3f}')
    # Paired bootstrap over problem groups, with fold repeats averaged within
    # each resample. Intervals describe test-case uncertainty, not new refits.
    ug,inv=np.unique(groups,return_inverse=True); rng=np.random.default_rng(1)
    bootidx=[]
    for _ in range(2000):
        sampled=rng.choice(len(ug),len(ug),replace=True)
        ii=np.concatenate([np.flatnonzero(inv==g) for g in sampled])
        if len(np.unique(y[ii]))==2: bootidx.append(ii)
    for base,full in [('competence','competence+seg'),('competence_quadratic','competence_quadratic+seg'),('competence_completion','competence_completion+seg')]:
        a,b=pred[base],pred[full]
        ac=(((b>=.5)==y).mean(0)-((a>=.5)==y).mean(0))
        avga,avgb=a.mean(0),b.mean(0)
        # Mean OOF scores for bootstrap AUC, explicitly not mean-repeat AUC.
        da=[auc(y[ii],avgb[ii])-auc(y[ii],avga[ii]) for ii in bootidx]
        dc=[ac[ii].mean() for ii in bootidx]
        print(f'{full} minus {base}: accuracy {ac.mean():+.3f} 95% group-bootstrap {np.quantile(dc,[.025,.975]).round(3)}; mean-score AUC {auc(y,avgb)-auc(y,avga):+.3f} CI {np.quantile(da,[.025,.975]).round(3)}')
    ci=[]
    for ii in bootidx:
        v=within_auc(y[ii],rate[ii],models[ii])
        if np.isfinite(v): ci.append(v)
    print('Within-model solve-rate AUC 95% problem-bootstrap CI:',np.quantile(ci,[.025,.975]).round(3))
    # Within-model ranks remove between-configuration means from association.
    p=pred['seg'].mean(0); rr=np.zeros(n); pr=np.zeros(n)
    for m,ii in mi.items():
        rr[ii]=rankdata(rate[ii])/len(ii); pr[ii]=rankdata(p[ii])/len(ii)
    print(f'Within-model rank correlation, solve rate vs fresh OOF SegProbe: {np.corrcoef(rr,pr)[0,1]:.3f}')
    np.savez(OUT/'organiser_competence_predictions.npz',y=y,groups=groups,models=models,rate=rate,rawseg=rawseg,**pred)
    for name,p in pred.items(): table['oof_'+name]=p.mean(0)
    table.to_csv(OUT/'organiser_competence_cases.csv')
    print('Caveats: eight draws estimate budget-limited solve rate, not latent competence; cached samples also feed TransBlackBox; references are locally curated rather than organiser-provided; exploratory analysis; subgroup Fisher/permutation tests ignore shared-problem dependence; bootstrap groups problems; added-feature tests are predictive, not causal mediation.')


if __name__=='__main__':
    main()
