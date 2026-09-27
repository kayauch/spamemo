// Routes share the same collision checks as walking. No straight line through furniture.
export function findRoomRoute(nav,start,target){
  if(!nav||![start.x,start.y,target.x,target.y].every(Number.isFinite)||!nav.valid(start.x,start.y))return null;
  const xs=nav.footprint.map(p=>p[0]),ys=nav.footprint.map(p=>p[1]);
  const lo=[Math.min(...xs),Math.min(...ys)],hi=[Math.max(...xs),Math.max(...ys)];
  const step=Math.max(.16,Math.max(hi[0]-lo[0],hi[1]-lo[1])/90),w=Math.ceil((hi[0]-lo[0])/step),h=Math.ceil((hi[1]-lo[1])/step);
  const coords=i=>({x:lo[0]+(i%w+.5)*step,y:lo[1]+(Math.floor(i/w)+.5)*step,z:nav.floor+.04});
  const segment=(a,b)=>{const n=Math.max(1,Math.ceil(Math.hypot(a.x-b.x,a.y-b.y)/.04));for(let j=0;j<=n;j++)if(!nav.valid(a.x+(b.x-a.x)*j/n,a.y+(b.y-a.y)*j/n))return false;return true;};
  let root=-1,rootDistance=Infinity;
  const valid=new Uint8Array(w*h);
  for(let i=0;i<valid.length;i++){const p=coords(i);if(!nav.valid(p.x,p.y))continue;valid[i]=1;const d=Math.hypot(p.x-start.x,p.y-start.y);if(d<rootDistance&&d<step*2&&segment(start,p)){root=i;rootDistance=d;}}
  if(root<0)return null;
  const previous=new Int32Array(w*h).fill(-2),queue=[root];previous[root]=-1;
  let end=root,best=Infinity;
  for(let q=0;q<queue.length;q++){
    const i=queue[q],p=coords(i),d=Math.hypot(p.x-target.x,p.y-target.y);if(d<best){best=d;end=i;}
    const x=i%w,y=Math.floor(i/w);
    for(const [dx,dy] of [[0,1],[1,0],[0,-1],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]){
      const nx=x+dx,ny=y+dy;if(nx<0||nx>=w||ny<0||ny>=h)continue;const j=ny*w+nx;
      if(!valid[j]||previous[j]!==-2||!segment(p,coords(j)))continue;
      previous[j]=i;queue.push(j);
    }
  }
  // A remote/disconnected target is not advertised as reachable.
  if(best>1.5)return null;
  const chain=[];for(let i=end;i!==-1;i=previous[i])chain.push(coords(i));chain.reverse();
  const path=[{x:start.x,y:start.y,z:nav.floor+.04},...chain];
  const simplified=[path[0]];let i=0;
  while(i<path.length-1){let j=path.length-1;while(j>i+1&&!segment(path[i],path[j]))j--;simplified.push(path[j]);i=j;}
  return simplified;
}

export function finitePosition(value){return !!value&&['x','y','z'].every(k=>Number.isFinite(value[k]));}
