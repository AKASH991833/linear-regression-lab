module.exports.run = async (page, base, shot) => {
  const out = [];
  const eq0 = await page.evaluate(()=>document.querySelector('#eq').textContent);
  out.push(['equation with preset', eq0.trim().slice(0,60)]);
  const r2 = await page.evaluate(()=>document.querySelector('#stR2').textContent);
  out.push(['R2', r2]);
  // clear -> empty state
  await page.evaluate(()=>document.querySelector('#clearBtn').click());
  const eqEmpty = await page.evaluate(()=>document.querySelector('#eq').textContent);
  out.push(['after clear', eqEmpty.trim()]);
  // add points by clicking canvas at 3 spots
  const cv = await page.$('#cv'); const bb = await cv.boundingBox();
  const pts=[[0.2,0.3],[0.5,0.55],[0.8,0.8],[0.35,0.42]];
  for(const [fx,fy] of pts){ await page.mouse.click(bb.x+bb.width*fx, bb.y+bb.height*fy); }
  await new Promise(r=>setTimeout(r,200));
  const n = await page.evaluate(()=>document.querySelector('#stN').textContent);
  const eq2 = await page.evaluate(()=>document.querySelector('#eq').textContent);
  out.push(['after adding 4 pts', n, eq2.trim().slice(0,50)]);
  // gradient descent steps
  await page.evaluate(()=>{for(let i=0;i<5;i++)document.querySelector('#gdStep').click()});
  const gds = await page.evaluate(()=>document.querySelector('#gdStatus').textContent);
  out.push(['gd status', gds.trim().slice(0,70)]);
  // quadratic
  await page.evaluate(()=>{[...document.querySelectorAll('#degSeg button')].find(b=>b.textContent==='Quadratic').click()});
  const eq3 = await page.evaluate(()=>document.querySelector('#eq').textContent);
  out.push(['quadratic eq', eq3.trim().slice(0,60)]);
  await page.screenshot({path:shot});
  return out;
};
