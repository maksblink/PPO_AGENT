import pandas as pd
import numpy as np
import pytest
from train_and_eval.dashboard.path_improvement import path_improvements


def chain(values, root=1):
    return pd.DataFrame({'run_id':range(root,root+len(values)), 'run.name':[f'r{i}' for i in range(len(values))],
        'path.root_run_id':[root]*len(values), 'path.parent_run_id':[None]+list(range(root,root+len(values)-1)),
        'path.status':['Complete']*len(values), 'analysis.stage':range(len(values)), 'eval.agent_return':values})


def test_histogram_all_bins_ties_and_deteriorations():
    f=pd.concat([chain([0,1,2,3,4]),chain([0,1,1,0,2],10)],ignore_index=True)
    h,d=path_improvements(f,f,'eval.agent_return')
    assert h.improved.tolist()==['4/4','3/4','2/4','1/4','0/4']
    assert h.paths.tolist()==[1,0,1,0,0]
    assert h.share_percent.sum()==100
    assert d.iloc[1]['ties']==1 and d.iloc[1]['deteriorations']==1


def test_no_bridging_filtered_stage():
    f=chain([0,1,2,3,4]);h,d=path_improvements(f.iloc[[0,4]],f,'eval.agent_return')
    assert h.empty and d.iloc[0]['comparisons']==0
    assert d.iloc[0]['filtered_parent_edges']==1


def test_missing_nonfinite_edges_not_zero_or_improvement():
    f=chain([0,1,np.nan,np.inf,3,4]);h,d=path_improvements(f,f,'eval.agent_return')
    assert d.iloc[0]['comparisons']==2 and d.iloc[0]['missing_metric_edges']==3
    assert h.iloc[0]['improved']=='2/2'


def test_different_denominators_separate():
    f=pd.concat([chain([1,2,3]),chain([3,2],10)],ignore_index=True)
    h,d=path_improvements(f,f,'eval.agent_return')
    assert set(h.comparisons)=={1,2}
    assert h.groupby('comparisons').share_percent.sum().tolist()==[100,100]


def test_decrease_and_drawdown_increase():
    f=chain([-.4,-.2,-.1])
    _,d=path_improvements(f,f,'eval.agent_return','Increase');assert d.iloc[0].improvements==2
    _,d=path_improvements(f,f,'eval.agent_return','Decrease');assert d.iloc[0].deteriorations==2


def test_branches_detected_even_if_filtered_out():
    f=chain([0,1,2]);branch=f.iloc[[2]].assign(run_id=4,**{'path.parent_run_id':1})
    full=pd.concat([f,branch],ignore_index=True)
    h,d=path_improvements(f,full,'eval.agent_return')
    assert h.empty and d.iloc[0]['status']=='Branched path (excluded)'


def test_unresolved_and_single_run_reported():
    f=chain([1]);h,d=path_improvements(f,f,'eval.agent_return');assert h.empty
    f['path.status']='Missing ancestor';f['path.root_run_id']=pd.NA
    h,d=path_improvements(f,f,'eval.agent_return');assert d.iloc[0]['status']=='Unresolved ancestry'


def test_empty_and_invalid_direction():
    f=chain([1]).iloc[:0];h,d=path_improvements(f,f,'eval.agent_return');assert h.empty and d.empty
    with pytest.raises(ValueError):path_improvements(f,f,'eval.agent_return','wrong')


def test_ui_distribution_changes_with_direction():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('''
import pandas as pd
from train_and_eval.dashboard.path_improvement import render_path_improvements
f=pd.DataFrame({'run_id':[1,2,3], 'run.name':['fresh','r1','r2'],
 'path.root_run_id':[1,1,1], 'path.parent_run_id':[None,1,2],
 'path.status':['Complete']*3, 'analysis.stage':[0,1,2], 'eval.agent_return':[.1,.2,.3]})
render_path_improvements(f,f,['eval.agent_return'],'TRAIN')
''').run()
    assert not app.exception
    assert app.dataframe[0].value['paths'].tolist()==[1,0,0]
    app.radio(key='ga_path_direction').set_value('Decrease').run()
    assert not app.exception
    assert app.dataframe[0].value['paths'].tolist()==[0,0,1]
