import numpy as np,json,time,itertools,sklearn
from sklearn.datasets import load_breast_cancer,make_classification
from sklearn.model_selection import StratifiedKFold,cross_val_score,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score

def model():return make_pipeline(StandardScaler(),SVC(C=1,gamma='scale'))
def score(X,y,S,cv):return 2*cross_val_score(model(),X[:,S],y,cv=cv,scoring='balanced_accuracy',n_jobs=1)-1

def greedy(X,y,cv,tau=.09):
 S=[];hist=[]
 for k in range(X.shape[1]):
  candidates=[]
  for p in range(X.shape[1]):
   if p not in S:
    v=score(X,y,S+[p],cv);candidates.append((v.mean(),-p,v))
  m,negp,v=max(candidates,key=lambda a:(a[0],a[1]));S.append(-negp)
  hist.append(dict(S=S.copy(),mean=float(m),std=float(v.std(ddof=1))))
  if k:
   den=np.hypot(hist[-1]['std'],hist[-2]['std']);delta=abs(m-hist[-2]['mean']);r=delta/den if den else (0 if delta==0 else float('inf'))
   hist[-1]['r']=r
   if r<tau:break
 best=max(range(len(hist)),key=lambda i:hist[i]['mean'])
 return hist[best]['S'],hist
D=load_breast_cancer();X=D.data;y=1-D.target
outer=StratifiedKFold(3,shuffle=True,random_state=42);rows=[];traces=[]
for fold,(tr,te) in enumerate(outer.split(X,y),1):
 A,b=X[tr],y[tr];cv=list(StratifiedKFold(3,shuffle=True,random_state=42).split(A,b))
 for name in ['All','Greedy','RFE','LASSO','RandomForest']:
  start=time.perf_counter()
  if name=='All':S=list(range(30))
  elif name=='Greedy':S,h=greedy(A,b,cv);traces.append(h)
  elif name=='RFE':
   z=StandardScaler().fit_transform(A);sel=RFE(SVC(kernel='linear',C=1),n_features_to_select=6,step=1).fit(z,b);S=np.flatnonzero(sel.support_).tolist()
  elif name=='LASSO':
   z=StandardScaler().fit_transform(A);m=Lasso(alpha=.01,max_iter=20000).fit(z,b);S=np.flatnonzero(abs(m.coef_)>1e-8).tolist()
   if not S:raise RuntimeError('empty selection')
  else:
   m=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=1).fit(A,b);S=np.argsort(-m.feature_importances_)[:6].tolist()
  elapsed=time.perf_counter()-start
  pred=model().fit(A[:,S],b).predict(X[te][:,S]);rows.append(dict(fold=fold,method=name,k=len(S),TSS=2*balanced_accuracy_score(y[te],pred)-1,F1=f1_score(y[te],pred),seconds=elapsed,S=S))
 print('fold',fold,flush=True)
A,b=make_classification(n_samples=240,n_features=8,n_informative=3,n_redundant=0,n_repeated=0,shuffle=False,random_state=42)
tr,te=train_test_split(np.arange(len(b)),test_size=.3,stratify=b,random_state=42);cv=list(StratifiedKFold(3,shuffle=True,random_state=42).split(A[tr],b[tr]))
cache={}
for k in range(1,9):
 for S in itertools.combinations(range(8),k):cache[S]=float(score(A[tr],b[tr],list(S),cv).mean())
path=[];compare=[]
for k in range(1,9):
 p=max((p for p in range(8) if p not in path),key=lambda p:(cache[tuple(sorted(path+[p]))],-p));path.append(p)
 opt=max((s for s in cache if len(s)==k),key=lambda s:cache[s]);g=tuple(sorted(path));pg=model().fit(A[tr][:,g],b[tr]).predict(A[te][:,g]);po=model().fit(A[tr][:,opt],b[tr]).predict(A[te][:,opt]);compare.append(dict(k=k,greedy=list(g),optimal=list(opt),cv_g=cache[g],cv_opt=cache[opt],gap=cache[opt]-cache[g],test_g=2*balanced_accuracy_score(b[te],pg)-1,test_opt=2*balanced_accuracy_score(b[te],po)-1))
sensitivity=[]
for tau in [0,.03,.09,.2,.5]:
 S,h=greedy(A[tr],b[tr],cv,tau);sensitivity.append(dict(tau=tau,k=len(S),steps=len(h),score=max(r['mean'] for r in h)))
json.dump(dict(version=sklearn.__version__,rows=rows,traces=traces,exhaustive=compare,sensitivity=sensitivity,feature_names=D.feature_names.tolist()),open('output/results.json','w'),indent=2)
print('DONE',flush=True)
