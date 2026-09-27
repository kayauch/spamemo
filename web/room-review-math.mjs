const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
const sub=(a,b)=>a.map((x,i)=>x-b[i]);
export function wallBetween(camera,target,s){
 if(!s?.origin||!s.u||!s.v)return false;
 const u=s.u,v=s.v,n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
 const a=dot(sub(camera,s.origin),n),b=dot(sub(target,s.origin),n);
 if(a*b>=0||Math.abs(a-b)<1e-8)return false;
 const t=a/(a-b);if(t<=.001||t>=.999)return false;
 const p=sub(camera.map((x,i)=>x+t*(target[i]-x)),s.origin);
 const uu=dot(u,u),vv=dot(v,v),uv=dot(u,v),den=uu*vv-uv*uv;
 if(den<1e-10)return false;
 const x=(dot(p,u)*vv-dot(p,v)*uv)/den,y=(dot(p,v)*uu-dot(p,u)*uv)/den;
 const upright=Math.abs(v[2])>Math.hypot(v[0],v[1])*4;
 return x>=-.05&&x<=1.05&&(upright||(y>=-.05&&y<=1.05));
}
export function insideFootprint(p,polygon){
 if(!polygon?.length)return false;let inside=false;
 for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
  const a=polygon[i],b=polygon[j];
  if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])inside=!inside;
 }return inside;
}
const floorKinds=new Set(['bed','sofa','cabinet','wardrobe','locker','shelf','desk','table','coffee_table','chair','dining_chair','box']);
const solidKinds=new Set(['bed','sofa','cabinet','wardrobe','locker','shelf','box']);
const size=b=>b.max.map((x,i)=>x-b.min[i]);
const overlap=(a,b)=>a.max.map((x,i)=>Math.max(0,Math.min(x,b.max[i])-Math.max(a.min[i],b.min[i])));
export function placementIssues(boxes,floor,height=2.5){
 const tolerance=Math.max(.03,Math.abs(height)*.04),valid=boxes.filter(b=>b.min?.length===3&&b.max?.length===3&&[...b.min,...b.max].every(Number.isFinite)&&size(b).every(x=>x>0));
 const issues=[];
 for(const b of valid){
  if(!floorKinds.has(b.label))continue;
  const gap=b.min[2]-floor;
  if(gap < -tolerance)issues.push({kind:'below_floor',ids:[b.id]});
  else if(gap>tolerance){
   const supported=valid.some(a=>a.id!==b.id&&Math.abs(b.min[2]-a.max[2])<=tolerance&&overlap(a,b).slice(0,2).reduce((x,y)=>x*y,1)>size(b)[0]*size(b)[1]*.15);
   // A box inside a cabinet is intentional storage; its supporting shelf may be unseen.
   const contained=valid.some(a=>a.id!==b.id&&['cabinet','wardrobe','locker','shelf'].includes(a.label)&&b.min.every((x,i)=>x>=a.min[i]-tolerance)&&b.max.every((x,i)=>x<=a.max[i]+tolerance));
   if(!supported&&!contained)issues.push({kind:'floating',ids:[b.id]});
  }
 }
 for(let i=0;i<valid.length;i++)for(let j=i+1;j<valid.length;j++){
  const a=valid[i],b=valid[j];
  // Hollow desks/chairs and decorations cannot be checked with bounding boxes.
  if(!solidKinds.has(a.label)||!solidKinds.has(b.label))continue;
  const d=overlap(a,b),volume=d.reduce((x,y)=>x*y,1),small=Math.min(...[a,b].map(x=>size(x).reduce((y,z)=>y*z,1)));
  const storage=(a.label==='box'&&['cabinet','wardrobe','locker','shelf'].includes(b.label))||(b.label==='box'&&['cabinet','wardrobe','locker','shelf'].includes(a.label));
  if(!storage&&d.every(x=>x>tolerance)&&volume/small>.22)issues.push({kind:'overlap',ids:[a.id,b.id]});
 }
 return issues;
}
