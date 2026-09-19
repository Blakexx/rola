"""The warp timeline page: one CTA's eight warps window by window, the real run's cycles from the kernel's stamps and
the instructions each activity executed in the traced run, with a strip per scheduler for its tensor pipe. `render`
takes `{label: timeline}` where a timeline is `tools/warp_timeline.py`'s JSON, and returns one self-contained HTML
page. See docs/internals/tools/warp_timeline.md.
"""
from __future__ import annotations

import json

from warp_trace import KEYS

EVENTS = ["head", "head_words", "head_scans", "snapshot", "edges", "readout", "fold", "sweep",
          "fold.walk", "fold.fragment", "fold.wait", "fold.fill", "readout.tile", "readout.issue", "readout.wait",
          "readout.drain", "lead:fold.walk", "lead:fold.fragment", "lead:fold.wait", "lead:fold.fill",
          "lead:readout.tile", "lead:readout.issue", "lead:readout.wait", "lead:readout.drain"]


def compact(timeline: dict) -> list:
    """A timeline's warps as the page's arrays: per warp, per window, [event index, t0, t1, *KEYS, unmatched]."""
    out = []
    for w in timeline["warps"]:
        wins = []
        for win in w["windows"]:
            ivs = []
            for iv in win["ivs"]:
                mx = iv["mix"] or {}
                ivs.append([EVENTS.index(iv["ev"]) if iv["ev"] in EVENTS else -1, iv["t0"], iv["t1"]]
                           + [int(mx.get(k, 0)) for k in KEYS] + [0 if iv["mix"] else 1])
            wins.append({"len": win["len"], "ivs": ivs})
        out.append(wins)
    return out


def render(timelines: dict[str, dict]) -> str:
    data = json.dumps({"evs": EVENTS, "keys": list(KEYS), "cells": {k: compact(v) for k, v in timelines.items()},
                       "census": {k: v.get("census") for k, v in timelines.items()}}, separators=(",", ":"))
    return PAGE.replace("DATA_JSON", data)


PAGE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Carry Warp Timeline</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#f6f4ef;--panel:#fffdf8;--ink:#1d1b17;--ink2:#5b574e;--muted:#8b867a;--line:#dcd7cb;--accent:#0f6f8c;
 --burst:#2a9d8f;--burst2:#3f7fbf;--decide:#e08a3c;--decide2:#c96a2a;--wait:#b5b0a5;--lap:#8a5fc7;--lap2:#b48ad9;--idle:#e9e4d8;--pipe:#1f5f7a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15161a;--panel:#1d1f25;--ink:#ece9e1;--ink2:#b7b3a8;--muted:#7f7b72;--line:#33363e;--accent:#5fb8d6;
 --burst:#3fbfae;--burst2:#5a9be0;--decide:#f0a054;--decide2:#d97f3f;--wait:#5c5a55;--lap:#a98ae0;--lap2:#c9b3ee;--idle:#2a2c33;--pipe:#7fd0ea}}
:root[data-theme="dark"]{--bg:#15161a;--panel:#1d1f25;--ink:#ece9e1;--ink2:#b7b3a8;--muted:#7f7b72;--line:#33363e;--accent:#5fb8d6;
 --burst:#3fbfae;--burst2:#5a9be0;--decide:#f0a054;--decide2:#d97f3f;--wait:#5c5a55;--lap:#a98ae0;--lap2:#c9b3ee;--idle:#2a2c33;--pipe:#7fd0ea}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "IBM Plex Sans",system-ui,sans-serif;padding:20px 16px 48px}
h1{font-size:20px;font-weight:600;margin:0 0 4px;text-wrap:balance}.sub{color:var(--ink2);margin:0 0 14px;max-width:70ch}
.bar{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:0 0 12px}label{color:var(--ink2);font-size:13px}
select,button{font:inherit;font-size:13px;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
.wrap{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;overflow-x:auto}
canvas{display:block;width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin:10px 0 0;font-size:12px;color:var(--ink2)}.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:5px}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font:12px/1.4 "IBM Plex Mono",monospace;color:var(--ink);box-shadow:0 4px 16px rgba(0,0,0,.15);display:none;max-width:360px;z-index:5}
table{border-collapse:collapse;margin-top:14px;font-variant-numeric:tabular-nums;font-size:13px}th,td{text-align:right;padding:4px 10px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:500}
.note{color:var(--muted);font-size:12px;margin-top:10px;max-width:80ch}
</style></head><body>
<h1>Carry Warp Timeline</h1>
<p class="sub">One CTA of the carry kernel: the real run's cycles from the kernel's own stamps, each interval carrying the instructions the same activity executed in a traced run. Rows are the four schedulers, each with its two warps and a strip for its tensor pipe.</p>
<div class="bar"><label>cell <select id="cell"></select></label><label>window <select id="win"></select></label><label>beneath each warp <select id="cmp"></select></label><label>scheduler pipe strip <select id="pipeMode"><option value="hmma">HMMA issue estimate</option><option value="burst">either warp in a burst</option></select></label><button id="reset">reset zoom</button><span id="info" style="color:var(--ink2);font-size:13px"></span></div>
<div class="wrap"><canvas id="cv" width="1400" height="520"></canvas>
<div class="legend"><span><i class="sw" style="background:var(--burst)"></i>fold fragment run</span><span><i class="sw" style="background:var(--burst2)"></i>readout tile</span><span><i class="sw" style="background:var(--decide)"></i>walk / fill</span><span><i class="sw" style="background:var(--decide2)"></i>readout issue / drain</span><span><i class="sw" style="background:var(--wait)"></i>wait</span><span><i class="sw" style="background:var(--lap)"></i>head, words, scans</span><span><i class="sw" style="background:var(--lap2)"></i>snapshot, edges, sweep</span><span><i class="sw" style="background:var(--pipe)"></i>pipe strip: dark = fed</span></div></div>
<div id="tip"></div>
<div id="panel" class="wrap" style="display:none;margin-top:12px"><div class="bar"><b id="ptitle"></b><label><input type="checkbox" id="byline"> roll up by source line</label><span id="preasons" style="color:var(--ink2);font-size:13px"></span></div><div id="ptable" style="max-height:480px;overflow:auto"></div></div>
<div id="summary"></div>
<p class="note">Drag on the canvas to zoom, hover an interval for its cycles and instruction mix, click one for its activity's instructions with the profiler's stall samples (a whole launch's samples at each instruction, attributed to the activity by the traced run's share of its executions). Pipe strip: for each interval its HMMAs times 32.5 cycles spread evenly over the interval, summed over the scheduler's two warps and capped at one; an estimate, not a measurement. Cycles are the SM's counter; the stamps cost one store each.</p>
<script>
const D = DATA_JSON;
const EV=D.evs, K=D.keys; const ki=Object.fromEntries(K.map((k,i)=>[k,i+3]));
const cls=e=>{const n=EV[e]||'';if(n.endsWith('fold.fragment'))return 'burst';if(n.endsWith('readout.tile'))return 'burst2';if(/fold\.(walk|fill)$/.test(n))return 'decide';if(/readout\.(issue|drain)$/.test(n))return 'decide2';if(/wait$/.test(n))return 'wait';if(/^(head|head_words|head_scans)$/.test(n))return 'lap';return 'lap2'};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue('--'+v).trim();
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
const selC=document.getElementById('cell'),selW=document.getElementById('win'),selP=document.getElementById('pipeMode'),selX=document.getElementById('cmp');
for(const c of Object.keys(D.cells)){const o=document.createElement('option');o.value=c;o.textContent=c;selC.appendChild(o)}
{const o=document.createElement('option');o.value='';o.textContent='nothing';selX.appendChild(o);for(const c of Object.keys(D.cells)){const q=document.createElement('option');q.value=c;q.textContent=c;selX.appendChild(q)}}
let view={x0:0,x1:1},boxes=[];
function fillWins(){selW.innerHTML='';const n=D.cells[selC.value][0].length;for(let i=0;i<n;i++){const o=document.createElement('option');o.value=i;o.textContent=i;selW.appendChild(o)}selW.value=Math.min(10,n-1)}
function draw(){const warps=D.cells[selC.value],wi=+selW.value;const otherCell=D.cells[selX.value];const L=Math.max(...warps.map(w=>w[wi].len),...(otherCell?otherCell.map(w=>(w[wi]||{len:0}).len):[]));
 const W=cv.width,H=cv.height,left=90,top=24,rowH=34,gap=6,schedH=rowH*2+18+gap*3;boxes=[];
 ctx.fillStyle=css('panel');ctx.fillRect(0,0,W,H);ctx.font='12px IBM Plex Mono, monospace';
 const X=t=>left+(t/L-view.x0)/(view.x1-view.x0)*(W-left-10);
 ctx.fillStyle=css('ink2');for(let g=0;g<=10;g++){const t=(view.x0+(view.x1-view.x0)*g/10)*L;const x=X(t);ctx.fillStyle=css('line');ctx.fillRect(x,top-4,1,H-top-10);ctx.fillStyle=css('muted');ctx.fillText(Math.round(t/1000)+'K',x+2,12)}
 const pipeMode=selP.value;
 for(let s=0;s<4;s++){const y0=top+s*schedH;
  for(const [j,w] of [[0,s],[1,s+4]]){const y=y0+j*(rowH+gap);ctx.fillStyle=css('ink2');ctx.fillText('S'+s+' W'+w,8,y+rowH/2+4);
   const other=D.cells[selX.value],hTop=other?Math.round(rowH*0.58):rowH;
   for(const iv of warps[w][wi].ivs){const x0=X(iv[1]),x1=X(iv[2]);if(x1<left||x0>W)continue;const c=cls(iv[0]);ctx.fillStyle=css(c);ctx.fillRect(Math.max(left,x0),y,Math.max(1,Math.min(W,x1)-Math.max(left,x0)),hTop);
    if(iv[3+K.length])ctx.fillStyle='rgba(0,0,0,.25)',ctx.fillRect(Math.max(left,x0),y,Math.max(1,x1-x0),hTop);
    boxes.push({x0:Math.max(left,x0),x1:Math.min(W,x1),y0:y,y1:y+hTop,iv,w,cell:selC.value})}
   if(other&&other[w]&&other[w][wi]){const y2=y+hTop+2,h2=rowH-hTop-2;
    for(const iv of other[w][wi].ivs){const x0=X(iv[1]),x1=X(iv[2]);if(x1<left||x0>W)continue;ctx.fillStyle=css(cls(iv[0]));ctx.globalAlpha=0.8;ctx.fillRect(Math.max(left,x0),y2,Math.max(1,Math.min(W,x1)-Math.max(left,x0)),h2);ctx.globalAlpha=1;
     boxes.push({x0:Math.max(left,x0),x1:Math.min(W,x1),y0:y2,y1:y2+h2,iv,w,cell:selX.value})}}}
  const py=y0+2*(rowH+gap);ctx.fillStyle=css('ink2');ctx.fillText('S'+s+' pipe',8,py+13);
  const bins=W-left-10,occ=new Float32Array(bins);
  for(const w of [s,s+4])for(const iv of warps[w][wi].ivs){const n=EV[iv[0]]||'';let d=0;if(pipeMode==='hmma'){const h=iv[ki.hmma];if(!h)continue;d=Math.min(1,h*32.5/Math.max(1,iv[2]-iv[1]))}else{if(!(n.endsWith('fold.fragment')||n.endsWith('readout.tile')))continue;d=1}
   const b0=Math.floor(X(iv[1])-left),b1=Math.ceil(X(iv[2])-left);for(let b=Math.max(0,b0);b<Math.min(bins,b1);b++)occ[b]=Math.min(1,occ[b]+d)}
  for(let b=0;b<bins;b++){const v=occ[b];ctx.fillStyle=v>0?css('pipe'):css('idle');ctx.globalAlpha=v>0?0.25+0.75*v:1;ctx.fillRect(left+b,py,1,14);ctx.globalAlpha=1}
  const fed=occ.reduce((a,v)=>a+v,0)/bins;ctx.fillStyle=css('muted');ctx.fillText('fed '+(100*fed).toFixed(0)+'% of the view',W-150,py+12)}
 document.getElementById('info').textContent='window '+wi+': '+Math.round(L)+' cycles (longest warp)';summary(warps,wi,L)}
function summary(warps,wi,L){const acc={};let idle=[0,0,0,0];
 for(let w=0;w<8;w++)for(const iv of warps[w][wi].ivs){const n=EV[iv[0]]||'?';const a=acc[n]||(acc[n]={cyc:0,cnt:0,n:0,hmma:0,copy:0,ldsm:0,back:0});a.cyc+=iv[2]-iv[1];a.cnt++;a.n+=iv[ki.n];a.hmma+=iv[ki.hmma];a.copy+=iv[ki.copy];a.ldsm+=iv[ki.ldsm];a.back+=iv[ki.back]}
 const rows=Object.entries(acc).sort((a,b)=>b[1].cyc-a[1].cyc).map(([n,a])=>`<tr><td>${n}</td><td>${(a.cyc/8).toFixed(0)}</td><td>${(a.cnt/8).toFixed(1)}</td><td>${(a.cyc/a.cnt).toFixed(0)}</td><td>${(a.n/a.cnt).toFixed(0)}</td><td>${(a.hmma/a.cnt).toFixed(0)}</td><td>${a.hmma?(a.cyc/(a.hmma*32.5)).toFixed(2):''}</td><td>${(a.ldsm/a.cnt).toFixed(0)}</td><td>${(a.copy/a.cnt).toFixed(0)}</td><td>${(a.back/a.cnt).toFixed(0)}</td></tr>`).join('');
 document.getElementById('summary').innerHTML=`<table><thead><tr><th>activity</th><th>cycles a warp</th><th>events a warp</th><th>cycles an event</th><th>instr an event</th><th>HMMA an event</th><th>cycles / (HMMA x 32.5)</th><th>ldmatrix</th><th>copies</th><th>back branches</th></tr></thead><tbody>${rows}</tbody></table>`}
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),sx=cv.width/r.width,sy=cv.height/r.height,x=(e.clientX-r.left)*sx,y=(e.clientY-r.top)*sy;const b=boxes.find(b=>x>=b.x0&&x<=b.x1&&y>=b.y0&&y<=b.y1);
 if(!b){tip.style.display='none';return}const iv=b.iv,n=EV[iv[0]]||'?',cyc=iv[2]-iv[1],h=iv[ki.hmma];
 tip.innerHTML=`<b>${n}</b> warp ${b.w} · ${b.cell}<br>${cyc} cycles, ${iv[ki.n]} instructions${iv[3+K.length]?' (mix unmatched)':''}<br>HMMA ${h}${h?' → floor '+Math.round(h*32.5)+' alone, '+Math.round(h*65)+' paired ('+(cyc/(h*32.5)).toFixed(2)+'x alone)':''}<br>ldmatrix ${iv[ki.ldsm]} · copies ${iv[ki.copy]} · lds ${iv[ki.lds]} · sts ${iv[ki.sts]} · shfl ${iv[ki.shfl]}<br>bar ${iv[ki.bar]} · mbarrier ${iv[ki.mbar]} · red ${iv[ki.red]} · back branches ${iv[ki.back]}`;
 tip.style.display='block';tip.style.left=Math.min(window.innerWidth-380,e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px'});
cv.addEventListener('mouseleave',()=>tip.style.display='none');
let picked=null;
function panel(){const C=D.census[selC.value];const p=document.getElementById('panel');if(!C||picked===null){p.style.display='none';return}
 const lab=EV[picked]||'?';const rows=C.by_event[lab]||[];const ins=Object.fromEntries(C.instructions.map(r=>[r[0],r]));const R=C.reasons;
 let tot=0,byR=new Array(R.length).fill(0);const items=[];
 for(const [pc,n] of rows){const r=ins[pc];if(!r)continue;const share=r[5]?n/r[5]:0;const smp=r[6]*share;tot+=smp;const rs=r[7].map(x=>x*share);rs.forEach((x,i)=>byR[i]+=x);items.push({pc,line:r[2],top:r[3],text:r[1],cls:r[4],n,smp,rs})}
 const top=byR.map((x,i)=>[R[i],x]).filter(x=>x[1]>0).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([k,x])=>k+' '+(100*x/tot).toFixed(0)+'%').join(' · ');
 document.getElementById('ptitle').textContent=lab+': '+items.length+' instructions, '+Math.round(tot)+' stall samples attributed';document.getElementById('preasons').textContent=top;
 let list=items;if(document.getElementById('byline').checked){const g={};for(const it of items){const k=it.line;const a=g[k]||(g[k]={pc:'',line:k,top:it.top,text:'',cls:'',n:0,smp:0,rs:new Array(R.length).fill(0),cnt:0});a.n+=it.n;a.smp+=it.smp;it.rs.forEach((x,i)=>a.rs[i]+=x);a.cnt++;a.text=a.cnt+' instructions'}list=Object.values(g)}
 list.sort((a,b)=>b.smp-a.smp);
 const h=['<table><thead><tr><th>source (innermost)</th><th>from</th><th>SASS</th><th>class</th><th>executed in this activity</th><th>samples</th><th>share</th><th>top reason</th></tr></thead><tbody>'];
 for(const it of list.slice(0,300)){const i=it.rs.indexOf(Math.max(...it.rs));h.push(`<tr><td>${it.line}</td><td>${it.top}</td><td style="font:12px IBM Plex Mono,monospace;text-align:left;max-width:420px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${it.text.replace(/</g,'&lt;')}</td><td>${it.cls}</td><td>${it.n}</td><td>${it.smp.toFixed(0)}</td><td>${tot?(100*it.smp/tot).toFixed(1):0}%</td><td>${it.smp>0?R[i]+' '+(100*it.rs[i]/it.smp).toFixed(0)+'%':''}</td></tr>`)}
 h.push('</tbody></table>');document.getElementById('ptable').innerHTML=h.join('');p.style.display='block'}
cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),sx=cv.width/r.width,sy=cv.height/r.height,x=(e.clientX-r.left)*sx,y=(e.clientY-r.top)*sy;const b=boxes.find(b=>x>=b.x0&&x<=b.x1&&y>=b.y0&&y<=b.y1);if(b&&Math.abs(x-(drag??x))<8){picked=b.iv[0];panel()}});
document.getElementById('byline').onchange=panel;
let drag=null;cv.addEventListener('mousedown',e=>{const r=cv.getBoundingClientRect();drag=(e.clientX-r.left)*cv.width/r.width});
cv.addEventListener('mouseup',e=>{if(drag===null)return;const r=cv.getBoundingClientRect(),x=(e.clientX-r.left)*cv.width/r.width;const a=Math.min(drag,x),b=Math.max(drag,x);drag=null;if(b-a<8)return;const f=t=>view.x0+(t-90)/(cv.width-100)*(view.x1-view.x0);view={x0:Math.max(0,f(a)),x1:Math.min(1,f(b))};draw()});
document.getElementById('reset').onclick=()=>{view={x0:0,x1:1};draw()};
selC.onchange=()=>{fillWins();view={x0:0,x1:1};draw()};selW.onchange=draw;selP.onchange=draw;selX.onchange=draw;
fillWins();draw();
</script></body></html>'''
