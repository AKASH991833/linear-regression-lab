/* Linear Regression Lab: interactive OLS, polynomial fit, gradient descent */
(function(){
'use strict';
const cv=$('#cv'),ctx=cv.getContext('2d');
const lossCv=$('#lossCv'),lctx=lossCv.getContext('2d');
const W=cv.width,H=cv.height,PAD=54;
let pts=[],degree=1,dragI=-1;
let gd={a:0,b:0,running:false,iters:0,losses:[],timer:null};
const X2Px=x=>PAD+(x/10)*(W-2*PAD), Y2Py=y=>H-PAD-(y/10)*(H-2*PAD);
const Px2X=px=>(px-PAD)/(W-2*PAD)*10, Py2Y=py=>(H-PAD-py)/(H-2*PAD)*10;

/* ---------- math ---------- */
function polyFit(pts,d){
  const n=pts.length;if(n<d+1)return null;
  // normal equations with Gaussian elimination
  const m=d+1,A=[],Y=[];
  for(let r=0;r<m;r++){A.push([]);for(let c=0;c<m;c++){let s=0;for(const p of pts)s+=Math.pow(p.x,r+c);A[r].push(s)}
    let s=0;for(const p of pts)s+=p.y*Math.pow(p.x,r);Y.push(s)}
  for(let c=0;c<m;c++){
    let piv=c;for(let r=c+1;r<m;r++)if(Math.abs(A[r][c])>Math.abs(A[piv][c]))piv=r;
    if(Math.abs(A[piv][c])<1e-12)return null;
    [A[c],A[piv]]=[A[piv],A[c]];[Y[c],Y[piv]]=[Y[piv],Y[c]];
    for(let r=0;r<m;r++){if(r===c)continue;const f=A[r][c]/A[c][c];for(let k=c;k<m;k++)A[r][k]-=f*A[c][k];Y[r]-=f*Y[c]}
  }
  return Y.map((y,i)=>y/A[i][i]); // coeffs c0..cd
}
const applyC=(c,x)=>c.reduce((s,ci,i)=>s+ci*Math.pow(x,i),0);
function stats(c){
  const n=pts.length;if(!c||n<2)return null;
  const mean=pts.reduce((s,p)=>s+p.y,0)/n;
  let ssRes=0,ssTot=0,sxy=0,sx=0,sy=0,sx2=0,sy2=0;
  for(const p of pts){const r=p.y-applyC(c,p.x);ssRes+=r*r;ssTot+=(p.y-mean)**2;sx+=p.x;sy+=p.y;sxy+=p.x*p.y;sx2+=p.x*p.x;sy2+=p.y*p.y}
  const r=n>1?(n*sxy-sx*sy)/Math.sqrt(Math.max(1e-12,(n*sx2-sx*sx)*(n*sy2-sy*sy))):0;
  return{r2:ssTot<1e-12?1:Math.max(0,1-ssRes/ssTot),r,mse:ssRes/n,n};
}
function bandPoints(c){
  // 95% CI of mean response (linear only)
  if(!c||pts.length<3||degree!==1)return null;
  const n=pts.length,xs=pts.map(p=>p.x),xm=xs.reduce((a,b)=>a+b)/n;
  const sxx=xs.reduce((s,x)=>s+(x-xm)**2,0);if(sxx<1e-9)return null;
  const mse=pts.reduce((s,p)=>s+(p.y-applyC(c,p.x))**2,0)/(n-2);
  const se=x=>Math.sqrt(mse*(1/n+(x-xm)**2/sxx));
  return{xm,fn:x=>2.06*se(x)}; // t approx for moderate n
}

/* ---------- rendering ---------- */
function draw(){
  const dark=document.documentElement.dataset.theme==='dark';
  ctx.clearRect(0,0,W,H);
  const ink=dark?'#a3b1cf':'#4b5b7a',line=dark?'#1d2c4e':'#e3e9f4',acc=getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()||'#0d9488';
  // grid
  ctx.strokeStyle=line;ctx.lineWidth=1;ctx.fillStyle=ink;ctx.font='12px sans-serif';ctx.textAlign='center';
  for(let i=0;i<=10;i++){
    const x=X2Px(i),y=Y2Py(i);
    ctx.beginPath();ctx.moveTo(x,PAD);ctx.lineTo(x,H-PAD);ctx.stroke();
    ctx.beginPath();ctx.moveTo(PAD,y);ctx.lineTo(W-PAD,y);ctx.stroke();
    ctx.fillText(i,x,H-PAD+18);ctx.textAlign='right';ctx.fillText(i,PAD-8,y+4);ctx.textAlign='center';
  }
  ctx.strokeStyle=ink;ctx.lineWidth=1.6;
  ctx.beginPath();ctx.moveTo(PAD,PAD-6);ctx.lineTo(PAD,H-PAD);ctx.lineTo(W-PAD+6,H-PAD);ctx.stroke();
  const c=gd.running||gd.iters>0?[gd.b,gd.a]:polyFit(pts,degree);
  // band
  const band=bandPoints(c);
  if($('#showBand').checked&&band){
    ctx.fillStyle=acc+'22';ctx.beginPath();
    for(let px=PAD;px<=W-PAD;px+=4){const x=Px2X(px),y=applyC(c,x)+band.fn(x);px===PAD?ctx.moveTo(px,Y2Py(y)):ctx.lineTo(px,Y2Py(y))}
    for(let px=W-PAD;px>=PAD;px-=4){const x=Px2X(px),y=applyC(c,x)-band.fn(x);ctx.lineTo(px,Y2Py(y))}
    ctx.closePath();ctx.fill();
  }
  // residuals
  if($('#showResid').checked&&c){
    ctx.strokeStyle='#dc262688';ctx.lineWidth=1.4;ctx.setLineDash([4,4]);
    for(const p of pts){ctx.beginPath();ctx.moveTo(X2Px(p.x),Y2Py(p.y));ctx.lineTo(X2Px(p.x),Y2Py(applyC(c,p.x)));ctx.stroke()}
    ctx.setLineDash([]);
  }
  // fit curve
  if(c){
    ctx.strokeStyle=acc;ctx.lineWidth=3;ctx.beginPath();
    for(let px=PAD;px<=W-PAD;px+=3){const y=applyC(c,Px2X(px));px===PAD?ctx.moveTo(px,Y2Py(y)):ctx.lineTo(px,Y2Py(y))}
    ctx.stroke();
  }
  // points
  for(let i=0;i<pts.length;i++){
    const p=pts[i],x=X2Px(p.x),y=Y2Py(p.y);
    ctx.beginPath();ctx.arc(x,y,i===dragI?9:7,0,7);ctx.fillStyle=acc;ctx.fill();
    ctx.lineWidth=2.5;ctx.strokeStyle=dark?'#0a1224':'#fff';ctx.stroke();
  }
  // equation + stats
  const st=stats(c);
  if(c&&st){
    const terms=c.map((ci,i)=>{const v=Math.abs(ci).toFixed(3);if(i===0)return `${ci<0?'-':''}${v}`;return ` ${ci<0?'-':'+'} ${v}${i===1?'x':'x^'+i}`}).join('');
    $('#eq').innerHTML=gd.running||gd.iters>0?`ŷ = ${terms} <span style="color:var(--ink-faint);font-size:.8rem;font-weight:600">&nbsp;(gradient descent, step ${gd.iters})</span>`:`ŷ = ${terms}`;
    $('#stR2').textContent=st.r2.toFixed(3);$('#stR').textContent=st.r.toFixed(3);$('#stMSE').textContent=st.mse.toFixed(3);
  }else{
    $('#eq').textContent=pts.length<2?'Add at least 2 points to fit a line':'Need more distinct points';
    $('#stR2').textContent='-';$('#stR').textContent='-';$('#stMSE').textContent='-';
  }
  $('#stN').textContent=pts.length;
  drawLoss();
}
function drawLoss(){
  lctx.clearRect(0,0,lossCv.width,lossCv.height);
  const L=gd.losses;if(L.length<2){return}
  const dark=document.documentElement.dataset.theme==='dark';
  const max=Math.max(...L),min=Math.min(...L);
  lctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()||'#0d9488';
  lctx.lineWidth=2;lctx.beginPath();
  L.forEach((v,i)=>{const x=6+i/(L.length-1)*(lossCv.width-12);const y=lossCv.height-8-((v-min)/(max-min+1e-12))*(lossCv.height-16);i?lctx.lineTo(x,y):lctx.moveTo(x,y)});
  lctx.stroke();
  lctx.fillStyle=dark?'#6d7d9f':'#8494b3';lctx.font='11px sans-serif';
  lctx.fillText('loss '+L[L.length-1].toFixed(4),8,14);
}

/* ---------- gradient descent (linear) ---------- */
function gdStepOnce(){
  const n=pts.length;if(n<2)return;
  const lr=+$('#lr').value;
  let da=0,db=0;
  for(const p of pts){const e=(gd.b+gd.a*p.x)-p.y;da+=e*p.x;db+=e}
  gd.a-=lr*da/n*0.1;gd.b-=lr*db/n*0.1;
  gd.iters++;
  const mse=pts.reduce((s,p)=>s+((gd.b+gd.a*p.x)-p.y)**2,0)/n;
  gd.losses.push(mse);if(gd.losses.length>300)gd.losses.shift();
  $('#gdStatus').textContent=`Iteration ${gd.iters} · slope ${gd.a.toFixed(3)} · intercept ${gd.b.toFixed(3)} · MSE ${mse.toFixed(4)}`;
}
function gdToggle(){
  if(gd.running){clearInterval(gd.timer);gd.running=false;$('#gdBtn').textContent='Resume learning';draw();return}
  if(gd.iters===0){
    if(pts.length<2){toast('Add at least 2 points first');return}
    degree=1;setDeg(1);
    gd.a=(Math.random()-0.5)*2;gd.b=5;gd.losses=[];
  }
  gd.running=true;$('#gdBtn').textContent='Pause';
  gd.timer=setInterval(()=>{for(let k=0;k<3;k++)gdStepOnce();draw()},60);
}
$('#gdBtn').addEventListener('click',gdToggle);
$('#gdStep').addEventListener('click',()=>{
  if(gd.iters===0){if(pts.length<2){toast('Add at least 2 points first');return}gd.a=(Math.random()-0.5)*2;gd.b=5;gd.losses=[];degree=1;setDeg(1)}
  for(let k=0;k<10;k++)gdStepOnce();draw();
});
function gdReset(){clearInterval(gd.timer);gd={a:0,b:0,running:false,iters:0,losses:[]};$('#gdBtn').textContent='Start learning';$('#gdStatus').textContent='Idle. Exact solution is shown in the mean time.'}

/* ---------- interaction ---------- */
function evPos(e){
  const r=cv.getBoundingClientRect();
  const cx=(e.touches?e.touches[0].clientX:e.clientX)-r.left, cy=(e.touches?e.touches[0].clientY:e.clientY)-r.top;
  return{x:cx*W/r.width,y:cy*H/r.height};
}
function hit(p){for(let i=pts.length-1;i>=0;i--){const dx=X2Px(pts[i].x)-p.x,dy=Y2Py(pts[i].y)-p.y;if(dx*dx+dy*dy<260)return i}return -1}
function down(e){
  const p=evPos(e);const i=hit(p);
  if(i>=0){dragI=i}else{pts.push({x:Math.min(10,Math.max(0,Px2X(p.x))),y:Math.min(10,Math.max(0,Py2Y(p.y)))});gdReset()}
  draw();e.preventDefault();
}
function move(e){if(dragI<0)return;const p=evPos(e);pts[dragI]={x:Math.min(10,Math.max(0,Px2X(p.x))),y:Math.min(10,Math.max(0,Py2Y(p.y)))};gdReset();draw();e.preventDefault()}
function up(){dragI=-1}
cv.addEventListener('mousedown',down);addEventListener('mousemove',move);addEventListener('mouseup',up);
cv.addEventListener('touchstart',down,{passive:false});cv.addEventListener('touchmove',(e)=>{move(e)},{passive:false});cv.addEventListener('touchend',up);
cv.addEventListener('dblclick',e=>{const i=hit(evPos(e));if(i>=0){pts.splice(i,1);gdReset();draw()}});

/* ---------- controls ---------- */
$('#showResid').addEventListener('change',draw);
$('#showBand').addEventListener('change',draw);
function setDeg(d){degree=d;$$('#degSeg button').forEach(b=>b.classList.toggle('on',+b.dataset.d===d));gdReset();draw()}
$('#degSeg').addEventListener('click',e=>{const b=e.target.closest('button');if(b)setDeg(+b.dataset.d)});
$('#clearBtn').addEventListener('click',()=>{pts=[];gdReset();draw()});
$('#lr').addEventListener('input',()=>$('#lrVal').textContent=(+$('#lr').value).toFixed(3));
const PRESETS={
  'noisy line':()=>Array.from({length:22},()=>{const x=Math.random()*10;return{x,y:Math.min(10,Math.max(0,1.5+0.7*x+(Math.random()-0.5)*2.4))}}),
  'strong curve':()=>Array.from({length:26},(_,i)=>{const x=i/2.6;return{x,y:Math.min(10,0.12*x*x+0.4+Math.random()*0.5)}}),
  'with outliers':()=>Array.from({length:18},()=>{const x=Math.random()*10;return{x,y:Math.random()<0.2?Math.random()*10:Math.min(10,Math.max(0,2+0.6*x+(Math.random()-0.5)))}}),
  'two clusters':()=>[...Array.from({length:9},()=>({x:1+Math.random()*2,y:1.5+Math.random()*2})),...Array.from({length:9},()=>({x:7+Math.random()*2,y:7+Math.random()*2}))],
};
let presetKeys=Object.keys(PRESETS),presetI=0;
$('#presetBtn').addEventListener('click',()=>{
  const k=presetKeys[presetI%presetKeys.length];presetI++;
  pts=PRESETS[k]();gdReset();$('#presetBtn').textContent='Load preset: '+presetKeys[presetI%presetKeys.length];draw();toast('Loaded: '+k);
});
$('#pngBtn').addEventListener('click',()=>{
  const a=document.createElement('a');a.download='regression-chart.png';a.href=cv.toDataURL('image/png');a.click();toast('Chart saved as PNG');
});
$('#csvBtn').addEventListener('click',()=>{
  const csv='x,y\n'+pts.map(p=>p.x.toFixed(4)+','+p.y.toFixed(4)).join('\n');
  const a=document.createElement('a');a.download='points.csv';a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.click();toast('CSV exported');
});
$('#csvIn').addEventListener('change',e=>{
  const f=e.target.files[0];if(!f)return;
  f.text().then(t=>{
    const rows=t.split(/\r?\n/).map(r=>r.split(/[,;\t]/).map(Number)).filter(r=>r.length>=2&&isFinite(r[0])&&isFinite(r[1]));
    if(!rows.length){toast('No numeric x,y rows found');return}
    pts=rows.map(r=>({x:Math.min(10,Math.max(0,r[0])),y:Math.min(10,Math.max(0,r[1]))})).slice(0,400);
    gdReset();draw();toast(`Imported ${pts.length} points`);
  });
});
new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
// seed with a friendly starter set
pts=PRESETS['noisy line']();presetI=1;draw();
})();
