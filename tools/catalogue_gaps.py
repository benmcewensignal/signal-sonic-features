"""The Catalogue gap report (site: data/catalogue-gaps.json).
Each scene's records this year, part by part, against the closest openly licensed loop in the worker's loop index
(tempo it can be fitted to; for bass and melody a key that mixes). "Far" means the best match scores under 0.45.
Basis: charting records where 60 or more have been split into parts, otherwise records released this year (the
union of data/scene-records-2026.json and the database's 2026 releases). Inputs: the Signal database, the separated
records' parts file (record-parts.npz) and worker/loop_index.pkl; paths below are the build machine's.
"""
import sys, json, sqlite3, pickle, collections, numpy as np
sys.path.insert(0,"/tmp/sf")
from worker.scene import _mixes
c=sqlite3.connect("/tmp/now.db")
chart=collections.defaultdict(dict)
for t,sc,rk in c.execute("select track_id, scene, chart_rank from track_scenes where chart_rank is not null and week like '2026-%'"):
    if rk and (t not in chart[sc] or rk<chart[sc][t]): chart[sc][t]=rk
released=collections.defaultdict(list)
for t,sc in c.execute("select distinct ts.track_id, ts.scene from track_scenes ts join track_meta m on m.track_id=ts.track_id where m.released like '2026-%'"):
    released[sc].append(t)
meta={t:(n,a) for t,n,a in c.execute("select track_id, name, artists from track_meta")}
for sc,rows in json.load(open("/tmp/sf/data/scene-records-2026.json"))["scenes"].items():   # the fuller released lists for the big scenes
    have=set(released[sc]); released[sc]+= [r[0] for r in rows if r[0] not in have]
RP=np.load("/tmp/record-parts.npz"); at={t:i for i,t in enumerate(RP["ids"].tolist())}; V=RP["V"].astype(np.float32); TEMPO=RP["tempo"]; KEY=RP["key"]
LI=pickle.load(open("/tmp/sf/worker/loop_index.pkl","rb"))
FAM={"drums":"drums","bass":"bass","other":"melody","vocals":"vocals"}; PI={"drums":0,"bass":1,"other":2,"vocals":3}
PLAIN={"crest":46,"centroid_hz":48,"flatness":50,"onsets_per_s":51}
WORDS={"centroid_hz":("brighter","darker"),"onsets_per_s":("busier","sparser"),"crest":("punchier","softer"),"flatness":("noisier","more tonal")}
def describe(rp, lp):
    out=[]
    for k,(up,dn) in WORDS.items():
        a=np.median([r[k] for r in rp if r.get(k)]) if rp else None; b=np.median([m[k] for m in lp if m.get(k)]) if lp else None
        if not a or not b: continue
        r=a/b
        if r>=1.2: out.append((np.log(r),up))
        elif r<=1/1.2: out.append((-np.log(r),dn))
    out.sort(reverse=True); return [w for _,w in out[:3]]
def artists(a):
    try: L=json.loads(a) if a and str(a).startswith("[") else [a]
    except Exception: L=[a]
    return ", ".join([x for x in L if x][:2])
res={"library":"Freesound (openly licensed: Creative Commons 0 and Attribution)","loops":{f:len(v["meta"]) for f,v in LI.items()},
     "basis_rule":"charting records this year where 60 or more have been split into parts; otherwise records released this year","scenes":{}}
for sc in set(chart)|set(released):
    tr=chart.get(sc,{}); ids=[t for t in tr if t in at]; basis="charting"
    if len(ids)<60:
        rel=[t for t in released.get(sc,[]) if t in at]
        if len(rel)>len(ids): ids=rel; basis="released"
    if len(ids)<15: continue
    S={"records":len(ids),"basis":basis,"parts":{}}
    for part,fam in FAM.items():
        L=LI.get(fam)
        if not L: continue
        Z=L["Z"].astype(np.float32); keep=L["keep"]; mu=L["mu"].astype(np.float32); sd=L["sd"].astype(np.float32)
        lt=np.array([m.get("tempo") or 0 for m in L["meta"]],np.float32); lk=[m.get("key") for m in L["meta"]]
        KM={}; best=[]; far=[]
        for t in ids:
            i=at[t]; v=V[i,PI[part]][keep]; z=(v-mu)/sd; z/=np.linalg.norm(z)+1e-9; sim=Z@z; T=float(TEMPO[i]) or 0
            ok=np.ones(len(lt),bool)
            if T:
                ok=np.zeros(len(lt),bool)
                for f in (1,2,0.5): ok|=(lt>0)&(np.abs(lt*f-T)/T<=0.08)
            kk=str(KEY[i])
            if fam in ("bass","melody") and kk:
                if kk not in KM: KM[kk]=np.array([(_mixes(kk,x) is not False) if x else True for x in lk])
                ok&=KM[kk]
            b=float(sim[ok].max()) if ok.any() else -1.0; best.append(b)
            if b<0.45: far.append(t)
        best=np.array(best)
        words=describe([{k:float(V[at[t],PI[part],j]) for k,j in PLAIN.items()} for t in far],[m.get("plain") or {} for m in L["meta"]])
        keys=collections.Counter(str(KEY[at[t]]) for t in far if str(KEY[at[t]])).most_common(3) if fam in ("bass","melody") else []
        ex=sorted(far,key=lambda t:(chart.get(sc,{}).get(t,9999), t))[:3]   # charting references first: recognisable
        S["parts"][part]={"close":round(float((best>=0.7).mean()),3),"some":round(float(((best>=0.45)&(best<0.7)).mean()),3),"far":round(float((best<0.45).mean()),3),
            "gap_words":words,"keys":[k for k,_ in keys],"examples":[{"name":(meta.get(t) or ("",""))[0],"artists":artists((meta.get(t) or ("",""))[1]),"id":t} for t in ex]}
    res["scenes"][sc]=S
old=json.load(open("/tmp/gaps_old.json"))
json.dump(res,open("/tmp/sg/data/catalogue-gaps.json","w"),separators=(",",":"))
b=collections.Counter(S["basis"] for S in res["scenes"].values())
print(f"  scenes: {len(res['scenes'])} ({dict(b)}) | records checked: {sum(S['records'] for S in res['scenes'].values()):,} | loops {res['loops']}")
print("  scenes under 60 records now:", [(k,S['records']) for k,S in res['scenes'].items() if S['records']<60])
for fam in ("drums","bass","other","vocals"):
    fr=[S["parts"][fam]["far"] for S in res["scenes"].values() if fam in S["parts"]]; print(f"  {fam:7}: nothing close for {np.mean(fr)*100:.0f} in 100 on average (range {min(fr)*100:.0f} to {max(fr)*100:.0f})")
for sc in ("hard-techno","amapiano","bass-house","140-deep-dubstep-grime"):
    S=res["scenes"].get(sc); o=old["scenes"].get(sc,{})
    print(f"  {sc}: {S['basis']} {S['records']} (was {o.get('basis')} {o.get('records')}) | bass far {S['parts']['bass']['far']*100:.0f} (was {o.get('parts',{}).get('bass',{}).get('far',0)*100:.0f}) | words {S['parts']['bass']['gap_words']}")
