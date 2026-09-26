"""Reproducible, leakage-aware campaign prioritization analysis. Run: python analysis.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'; OUT.mkdir(exist_ok=True)
DATA = ROOT / 'data' / 'bank-additional-full.csv'
FEATURES = ['age','job','marital','education','default','housing','loan','pdays','previous','poutcome']
NUM = ['age','pdays','previous']; CAT = [x for x in FEATURES if x not in NUM]
SEED = 42

def make_pipe(kind):
    numeric = Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())])
    categorical = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    prep = ColumnTransformer([('num',numeric,NUM),('cat',categorical,CAT)])
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED) if kind == 'logistic' else HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,learning_rate=.06,l2_regularization=2,random_state=SEED)
    return Pipeline([('prepare',prep),('model',model)])

def top_k(y,p,k=.2):
    n=max(1,int(np.ceil(len(y)*k))); idx=np.argsort(-np.asarray(p),kind='stable')[:n]
    observed=int(np.asarray(y)[idx].sum()); baseline=float(np.mean(y));
    return {'contacts':n,'subscribers_found':observed,'precision':float(observed/n),'overall_rate':baseline,'lift':float(observed/n/baseline) if baseline else None,'capture_rate':float(observed/sum(y)) if sum(y) else None}

def metric(y,p):
    return {'pr_auc':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p)),'top_20pct':top_k(y,p)}

def main():
    df=pd.read_csv(DATA,sep=';')
    assert len(df)==41188 and set(FEATURES+['y']).issubset(df.columns)
    assert df.y.isin(['yes','no']).all()
    y=df.y.eq('yes').astype(int)
    X=df[FEATURES].copy()
    # -1 means never contacted before; encode it explicitly, never treat it as a day count.
    X['pdays']=X['pdays'].replace(-1,np.nan)
    n=len(df); a=int(.7*n); b=int(.85*n)
    splits={'train':(0,a),'validation':(a,b),'test':(b,n)}
    scores={}; fitted={}
    for kind in ('logistic','hist_gradient_boosting'):
        m=make_pipe(kind).fit(X.iloc[:a],y.iloc[:a]); fitted[kind]=m
        pv=m.predict_proba(X.iloc[a:b])[:,1]; scores[kind]=metric(y.iloc[a:b],pv)
    winner=max(scores,key=lambda key: scores[key]['pr_auc'])
    # Explain model ranking on a fixed validation sample, never on the untouched test.
    rng=np.random.default_rng(SEED)
    sample_idx=np.sort(rng.choice(np.arange(a,b),size=min(1200,b-a),replace=False))
    importance=permutation_importance(fitted[winner],X.iloc[sample_idx],y.iloc[sample_idx],
                                      scoring='average_precision',n_repeats=3,random_state=SEED)
    feature_importance=sorted([{'feature':name,'ap_drop_mean':float(avg),'ap_drop_std':float(std)}
                               for name,avg,std in zip(FEATURES,importance.importances_mean,importance.importances_std)],
                              key=lambda item:item['ap_drop_mean'],reverse=True)
    (OUT/'permutation_importance.json').write_text(json.dumps({
        'method':'Validation-only grouped permutation importance (average precision loss)',
        'rows':len(sample_idx),'repeats':3,'features':feature_importance},indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,5))
    rank=list(reversed(feature_importance))
    ax.barh([item['feature'] for item in rank],[item['ap_drop_mean'] for item in rank],color='#73a89a')
    ax.axvline(0,color='#585254',linewidth=.8)
    ax.set(xlabel='Decrease in validation average precision after shuffling',title='Model reliance on available features')
    fig.tight_layout();fig.savefig(OUT/'05_feature_importance.png',dpi=160);plt.close(fig)
    # The model is saved for reproducible batch scoring, not promoted as deployable.
    joblib.dump({'model':fitted[winner],'features':FEATURES,'training_rows':a,
                 'selected_on':'validation average precision','seed':SEED},OUT/'model.joblib')
    # Winner selected solely on validation; the untouched test is evaluated once.
    m=fitted[winner]; p=m.predict_proba(X.iloc[b:])[:,1]; yt=y.iloc[b:].to_numpy()
    test=metric(yt,p)
    pval=m.predict_proba(X.iloc[a:b])[:,1]
    # Contact capacity fixed at top 20%, not a test-tuned threshold.
    threshold=float(np.sort(pval)[-max(1,int(np.ceil((b-a)*.2)))])
    yp=(p>=threshold).astype(int)
    results={'dataset_rows':n,'positive_total':int(y.sum()),'positive_rate':float(y.mean()),'splits':{k:{'rows':v-u,'subscribers':int(y.iloc[u:v].sum()),'rate':float(y.iloc[u:v].mean())} for k,(u,v) in splits.items()},'excluded_columns':sorted(set(df.columns)-set(FEATURES)-{'y'}),'validation':scores,'selected_model':winner,'test':test,'validation_top20_threshold':threshold,'test_at_validation_threshold':{'contacts':int(yp.sum()),'precision':float((yt*yp).sum()/max(yp.sum(),1)),'recall':float((yt*yp).sum()/max(yt.sum(),1)),'confusion_matrix':confusion_matrix(yt,yp).tolist()}}
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    # Reproducible, anonymized score output: no names or account identifiers in UCI data.
    pd.DataFrame({'source_row':np.arange(b,n),'actual_subscription':yt,'probability':p,'top_20pct_ranked':False}).assign(top_20pct_ranked=lambda z:z.index.isin(np.argsort(-p,kind='stable')[:int(np.ceil(len(z)*.2))])).to_csv(OUT/'test_scores.csv',index=False)
    sns.set_theme(style='whitegrid',palette=['#73a89a','#251f21'])
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    count=y.value_counts().sort_index(); ax[0].bar(['No','Yes'],count.values,color=['#585254','#73a89a']); ax[0].set(title='Subscription outcomes',ylabel='Records')
    rate=df.assign(subscribed=y).groupby('poutcome')['subscribed'].agg(['mean','count']).sort_values('mean'); ax[1].barh(rate.index,rate['mean'],color='#73a89a'); ax[1].set(title='Observed rate by previous outcome',xlabel='Subscribed / records'); fig.tight_layout(); fig.savefig(OUT/'01_overview.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4.2)); prec,rec,_=precision_recall_curve(yt,p); ax[0].plot(rec,prec,color='#286e61',label=winner);ax[0].axhline(yt.mean(),ls='--',color='#585254',label='Prevalence'); ax[0].set(xlabel='Recall',ylabel='Precision',title='Untouched holdout precision-recall',xlim=(0,1),ylim=(0,1));ax[0].legend()
    frac,mean=calibration_curve(yt,p,n_bins=10,strategy='quantile');ax[1].plot(mean,frac,'o-',color='#286e61');ax[1].plot([0,1],[0,1],'--',color='#585254');ax[1].set(xlabel='Mean predicted probability',ylabel='Observed subscription rate',title='Holdout calibration',xlim=(0,1),ylim=(0,1));fig.tight_layout();fig.savefig(OUT/'02_model_quality.png',dpi=160);plt.close(fig)
    cm=confusion_matrix(yt,yp); fig,ax=plt.subplots(figsize=(5,4.2));ConfusionMatrixDisplay(cm,display_labels=['No','Yes']).plot(ax=ax,colorbar=False,cmap='BuGn');ax.set_title('Holdout at validation-selected threshold');fig.tight_layout();fig.savefig(OUT/'03_operating_point.png',dpi=160);plt.close(fig)
    sample=pd.DataFrame({'score':p,'subscribed':yt}).sort_values('score',ascending=False).reset_index(drop=True);sample['decile']=np.minimum(sample.index*10//len(sample),9)+1
    rates=sample.groupby('decile').subscribed.mean();fig,ax=plt.subplots(figsize=(8,4));ax.bar(rates.index,rates.values,color='#73a89a');ax.axhline(yt.mean(),ls='--',color='#585254',label='Holdout average');ax.set(title='Observed subscription by score decile (1 = highest)',xlabel='Ranked score decile',ylabel='Subscription rate',xticks=range(1,11));ax.legend();fig.tight_layout();fig.savefig(OUT/'04_decile_lift.png',dpi=160);plt.close(fig)
    print(json.dumps({'selected_model':winner,'test':test,'test_at_validation_threshold':results['test_at_validation_threshold']},indent=2))

if __name__=='__main__': main()
