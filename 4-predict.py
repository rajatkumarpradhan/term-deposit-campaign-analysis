"""Score a CSV using the historical model. Exploratory use only, never automated outreach."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='CSV with pre-call features, semicolon or comma delimited')
    parser.add_argument('--output',required=True,help='Destination CSV for rankings')
    parser.add_argument('--capacity',type=float,default=.2,help='Fraction to select, >0 and <=1 (default .2)')
    args=parser.parse_args()
    if not 0 < args.capacity <= 1: parser.error('--capacity must be >0 and <=1')
    payload=joblib.load(ROOT/'outputs'/'model.joblib')
    src=Path(args.input).resolve();dest=Path(args.output).resolve()
    if src==dest: parser.error('--input and --output must differ')
    data=pd.read_csv(src,sep=None,engine='python')
    if data.empty: parser.error('input CSV is empty')
    missing=set(payload['features'])-set(data.columns)
    if missing: parser.error('missing features: '+', '.join(sorted(missing)))
    features=data[payload['features']].copy()
    features['pdays']=pd.to_numeric(features['pdays'],errors='coerce').replace(-1,np.nan)
    features['age']=pd.to_numeric(features['age'],errors='coerce')
    features['previous']=pd.to_numeric(features['previous'],errors='coerce')
    scores=payload['model'].predict_proba(features)[:,1]
    n=max(1,int(np.ceil(len(scores)*args.capacity)))
    rank=np.argsort(-scores,kind='stable')
    selected=np.zeros(len(scores),dtype=bool);selected[rank[:n]]=True
    output=pd.DataFrame({'source_row':np.arange(len(scores)),
                         'model_score_not_calibrated_probability':scores,
                         'rank':pd.Series(scores).rank(ascending=False,method='first').astype(int),
                         'selected_for_review':selected}).sort_values('rank')
    dest.parent.mkdir(parents=True,exist_ok=True)
    output.to_csv(dest,index=False)
    print(f'Ranked {len(scores)} records; flagged {n} for review. Historical model: not a live-contact or eligibility decision.')

if __name__=='__main__': main()
