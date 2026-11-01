"""Analytically labelled partition fixtures and extraction scores."""
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

def arial_font_path():
    candidates=[os.environ.get('PHASEGRAPH_ARIAL_FONT',''),'C:/Windows/Fonts/arial.ttf','/Library/Fonts/Arial.ttf','/usr/share/fonts/truetype/msttcorefonts/Arial.ttf']
    for path in candidates:
        if path and Path(path).is_file():return path
    raise FileNotFoundError('Arial is required for the annotation benchmark; set PHASEGRAPH_ARIAL_FONT')

def synth_fixture(seed,condition):
    rng=np.random.default_rng(seed);rects=[(40,40,600,440)]
    for _ in range(2+seed%8):
        candidates=[(i,(r[2]-r[0])*(r[3]-r[1])) for i,r in enumerate(rects) if max(r[2]-r[0],r[3]-r[1])>105]
        i=max(candidates,key=lambda x:x[1])[0];x1,y1,x2,y2=rects.pop(i)
        vertical=(x2-x1>y2-y1) if rng.random()<.7 else (x2-x1>100)
        if vertical:
            x=int(x1+(x2-x1)*rng.uniform(.30,.70));rects.extend([(x1,y1,x,y2),(x,y1,x2,y2)])
        else:
            y=int(y1+(y2-y1)*rng.uniform(.30,.70));rects.extend([(x1,y1,x2,y),(x1,y,x2,y2)])
    image=np.full((480,640,3),255,np.uint8);reference=np.zeros((480,640),np.int16)
    for k,(x1,y1,x2,y2) in enumerate(rects,1):
        reference[y1+2:y2-1,x1+2:x2-1]=k
        cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.rectangle(image,(40,40),(600,440),(0,0,0),2)
    expected=set();contacts=[]
    for i,r in enumerate(rects):
        for j,s in enumerate(rects[:i]):
            if (r[0]==s[2] or r[2]==s[0]) and min(r[3],s[3])-max(r[1],s[1])>0:
                expected.add((j+1,i+1));contacts.append((r[0] if r[0]==s[2] else r[2],(max(r[1],s[1])+min(r[3],s[3]))//2))
            elif (r[1]==s[3] or r[3]==s[1]) and min(r[2],s[2])-max(r[0],s[0])>0:
                expected.add((j+1,i+1));contacts.append(((max(r[0],s[0])+min(r[2],s[2]))//2,r[1] if r[1]==s[3] else r[3]))
    cx,cy=contacts[seed%len(contacts)]
    if condition in ['text','both']:
        pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(pil)
        draw.text((cx-25,cy-45),'α',font=ImageFont.truetype(arial_font_path(),72),fill='black')
        image=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
    if condition in ['arrow','both']:
        cv2.arrowedLine(image,(max(50,cx-75),max(55,cy-45)),(min(585,cx+30),min(425,cy+15)),(0,0,0),2,tipLength=.16)
    return image,reference,expected,rects

def synthetic_scores(pred,reference,expected,n_true):
    roi=pred['roi'];x1,y1,x2,y2=roi;truth=reference[y1:y2,x1:x2]
    pids=[r['label'] for r in pred['regions']];tids=list(range(1,n_true+1));iou=np.zeros((len(pids),n_true))
    for i,pi in enumerate(pids):
        a=pred['labels']==pi
        for j,tj in enumerate(tids):
            b=truth==tj;u=(a|b).sum();iou[i,j]=(a&b).sum()/u if u else 0
    pairs=linear_sum_assignment(-iou) if len(pids) else ([],[])
    matched={pids[i]:tids[j] for i,j in zip(*pairs) if iou[i,j]>=.5}
    converted=[]
    for e in pred['edges']:
        u=matched.get(e['u_label'],-e['u_label']);v=matched.get(e['v_label'],-e['v_label'])
        converted.append(tuple(sorted((u,v))))
    got=set(converted);tp=len(got&expected);fp=len(got-expected);fn=len(expected-got)
    return {'n_reference_regions':n_true,'n_predicted_regions':len(pids),'region_count_absolute_error':abs(len(pids)-n_true),
        'node_precision':len(matched)/len(pids) if pids else 0,'node_recall':len(matched)/n_true,
        'node_F1':2*len(matched)/(len(pids)+n_true),'edge_precision':tp/(tp+fp) if tp+fp else 0,
        'edge_recall':tp/(tp+fn) if tp+fn else 0,'edge_F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
        'exact_graph_recovery':len(matched)==n_true==len(pids) and got==expected}
