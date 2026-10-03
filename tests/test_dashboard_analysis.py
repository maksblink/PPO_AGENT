import numpy as np
import pandas as pd
import pytest
from train_and_eval.dashboard.analysis import prepare_analysis, filter_rows, summarize, evaluation_metrics


def sample():
    return prepare_analysis(pd.DataFrame({
        'run_id':[1,2,3,4], 'path.root_run_id':[1,1,3,1],
        'path.parent_run_id':[None,1,None,1], 'path.position':[1,2,1,2],
        'path.status':['Complete']*4, 'run.status':['completed']*3+['failed'],
        'lr':[.000015,.000025,.00005,.000025], 'gamma':[.9,.9,.95,.9],
        'eval.agent_return':[.1,.2,np.nan,np.inf], 'eval.id':[11,12,None,None]}))


def test_stages_and_branches_use_full_lineage():
    f=sample()
    assert f['analysis.stage'].tolist()==[0,1,0,1]
    assert f.loc[f['analysis.latest_existing'],'run_id'].tolist()==[2,3,4]
    assert filter_rows(f,{}, {'lr':(.000015,.000015)},latest=True).empty


def test_ranges_inclusive_and_combined_values():
    f=sample()
    result=filter_rows(f,{'gamma':[.9]}, {'lr':(.000015,.000025)},stages=[0,1])
    assert result.run_id.tolist()==[1,2,4]
    assert filter_rows(f,{'gamma':[]},{}).empty
    assert filter_rows(f,{}, {},stages=[]).empty
    with pytest.raises(ValueError):filter_rows(f,{}, {'lr':(.1,.01)})


def test_stats_finite_counts_and_equal_run_weight():
    result=summarize(sample(),[],['eval.agent_return']).iloc[0]
    assert result['runs']==4 and result['paths']==2 and result['N']==2
    assert result['missing_or_nonfinite']==2
    assert result['mean']==pytest.approx(.15)
    assert result['median']==pytest.approx(.15)
    assert result['min']==.1 and result['max']==.2


def test_combined_groups_and_missing_group_preserved():
    f=sample(); f.loc[0,'gamma']=np.nan
    result=summarize(f,['gamma','lr'],['eval.agent_return'])
    assert result['runs'].sum()==4
    assert result['gamma'].isna().sum()==1


def test_missing_metric_not_zero():
    result=summarize(sample().iloc[[2]],[],['eval.agent_return']).iloc[0]
    assert result['N']==0 and pd.isna(result['mean'])


def test_duplicate_runs_rejected():
    f=sample()
    with pytest.raises(ValueError):summarize(pd.concat([f,f]),[],['eval.agent_return'])


def test_identifiers_not_metrics():
    assert evaluation_metrics(sample())==['eval.agent_return']


def test_unresolved_lineage_not_latest():
    f=sample();f.loc[2,'path.status']='Missing ancestor'
    assert not prepare_analysis(f).loc[2,'analysis.latest_existing']
