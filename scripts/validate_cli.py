"""Exercise the shipped wheel, not an accidental editable-source import."""
from pathlib import Path
import argparse, json, subprocess, sys, unittest, re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;PACKAGE=PROJECT
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=PROJECT/'output/cli_integration')
OUT=parser.parse_args().out
if OUT.exists() and any(OUT.iterdir()):parser.error('Choose a fresh --out directory for the integration rerun')
OUT.mkdir(parents=True,exist_ok=True)
installed=PROJECT/'build/wheel_install'
launcher="import sys;sys.path.insert(0,sys.argv[1]);import phasegraph;assert sys.argv[1] in phasegraph.__file__;from phasegraph.cli import main;main(sys.argv[2:])"
steps=[
    ['doctor'],
    ['extract','--images',str(PACKAGE/'examples/images'),'--out',str(OUT/'extract')],
    ['matrix','--graphs',str(OUT/'extract'),'--out',str(OUT/'matrix')],
    ['fit','--graphs',str(OUT/'extract'),'--out',str(OUT/'fit')],
    ['cluster','--matrix',str(OUT/'matrix/distance_matrix.csv'),'--groups','2','--out',str(OUT/'cluster')],
    ['query','--model',str(OUT/'fit/model.json'),'--graph',str(OUT/'extract/fixture_48/graph.json'),'--top-k','3','--out',str(OUT/'query')],
    ['extract','--images',str(PACKAGE/'examples/images'),'--canonical-width','800','--out',str(OUT/'canonical_extract')],
    ['fit','--graphs',str(OUT/'extract'),'--overflow','tail','--out',str(OUT/'tail_fit')],
    ['extract','--images',str(PACKAGE/'examples/images'),'--coordinate-metric','axes','--out',str(OUT/'axes_extract')],
]
records=[]
for args in steps:
    run=subprocess.run([sys.executable,'-I','-c',launcher,str(installed),*args],capture_output=True,text=True,check=True)
    records.append({'command':args,'exit_code':run.returncode,'stdout':run.stdout})
    if args[0]!='doctor':
        folder=Path(args[args.index('--out')+1])
        assert all((folder/leaf).exists() for leaf in ['command.txt','config.yaml','result.csv','run.log','run.json'])
        assert json.loads((folder/'run.json').read_text())['status']=='complete'
query=pd.read_csv(OUT/'query/result.csv');assert query.iloc[0]['name']=='fixture_48' and query.iloc[0].distance<1e-7
a=pd.read_csv(OUT/'matrix/distance_matrix.csv',index_col=0).values
b=pd.read_csv(OUT/'fit/distance_matrix.csv',index_col=0).values
assert np.array_equal(a,b)
unit_launcher="import sys,unittest;sys.path.insert(0,sys.argv[1]);import phasegraph;assert sys.argv[1] in phasegraph.__file__;suite=unittest.defaultTestLoader.discover(sys.argv[2]);r=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not r.wasSuccessful())"
unit=subprocess.run([sys.executable,'-I','-c',unit_launcher,str(installed),str(PACKAGE/'tests')],capture_output=True,text=True,check=True)
(OUT/'contract_tests.log').write_text(unit.stdout+unit.stderr,encoding='utf-8')
count=int(re.search(r'Ran (\d+) tests',unit.stdout+unit.stderr).group(1))
from phasegraph import read_graph,write_graph
long=read_graph(OUT/'extract/fixture_48/graph.json');long.graph['name']='long_query'
for _,_,a in long.edges(data=True):a['edge_distance']*=100
write_graph(long,OUT/'long_query.json')
args=['query','--model',str(OUT/'tail_fit/model.json'),'--graph',str(OUT/'long_query.json'),'--top-k','3','--out',str(OUT/'tail_query')]
run=subprocess.run([sys.executable,'-I','-c',launcher,str(installed),*args],capture_output=True,text=True,check=True)
assert pd.read_csv(OUT/'tail_query/result.csv').query_overflow.all()
assert all(read_graph(path).graph['processing_size'][0]==800 for path in (OUT/'canonical_extract').glob('*/graph.json'))
assert all(read_graph(path).graph['coordinate_metric']=='axes' for path in (OUT/'axes_extract').glob('*/graph.json'))
records.append({'command':args,'exit_code':run.returncode,'stdout':run.stdout})
hypotheses=OUT/'query_hypotheses';hypotheses.mkdir()
for i,cost_scale in enumerate([1.,1.01]):
    sample=read_graph(OUT/'extract/fixture_48/graph.json');sample.graph['name']='query_hypothesis_'+str(i)
    for _,_,attrs in sample.edges(data=True):attrs['edge_distance']*=cost_scale
    write_graph(sample,hypotheses/(str(i)+'.json'))
args=['query','--model',str(OUT/'tail_fit/model.json'),'--hypotheses',str(hypotheses),'--top-k','1','--out',str(OUT/'stable_query')]
run=subprocess.run([sys.executable,'-I','-c',launcher,str(installed),*args],capture_output=True,text=True,check=True)
stable=json.loads((OUT/'stable_query/stability.json').read_text())
assert stable['status']=='complete' and stable['hypotheses']==2 and stable['nominal_top_k']==['fixture_48']
assert all((OUT/'stable_query'/leaf).exists() for leaf in ['command.txt','config.yaml','result.csv','run.log','run.json'])
records.append({'command':args,'exit_code':run.returncode,'stdout':run.stdout})
for args in [
    ['fit','--graphs',str(OUT/'extract'),'--encoding','continuous','--overflow','tail','--out',str(OUT/'continuous_fit')],
    ['query','--model',str(OUT/'continuous_fit/model.json'),'--hypotheses',str(hypotheses),'--nominal-hypothesis','query_hypothesis_0','--top-k','1','--out',str(OUT/'continuous_query')],
]:
    run=subprocess.run([sys.executable,'-I','-c',launcher,str(installed),*args],capture_output=True,text=True,check=True)
    records.append({'command':args,'exit_code':run.returncode,'stdout':run.stdout})
    destination=Path(args[args.index('--out')+1])
    assert all((destination/leaf).exists() for leaf in ['command.txt','config.yaml','result.csv','run.log','run.json'])
assert json.loads((OUT/'continuous_fit/model.json').read_text())['encoding']=='continuous'
assert json.loads((OUT/'continuous_query/stability.json').read_text())['nominal_hypothesis']==0
(OUT/'audit.json').write_text(json.dumps({'wheel_import_verified':True,'commands_passed':len(records),'distinct_commands_passed':6,'contract_tests_passed':count,'self_retrieval_distance':query.iloc[0].distance,'runs':records},indent=2),encoding='utf-8')
print(f'Verified shipped wheel: {len(records)} executions of all six commands; {count} contracts pass.')
